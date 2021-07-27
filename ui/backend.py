import logging
import os
import sys
import uuid
from datetime import datetime

from flask import Flask, jsonify, request
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from oracle.utils import *
from utils.video_reader import DecordVideoReader
import numpy as np
import re
import math
from datetime import timedelta

from logging.config import dictConfig
from PIL import Image

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'DEBUG',
        'handlers': ['wsgi']
    }
})

logging.basicConfig(level=logging.DEBUG)
appdata_url_path = "/appdata"
appdata_folder = ".."
app = Flask(__name__,
            static_url_path=appdata_url_path,
            static_folder=appdata_folder)

# video_path = "/mnt/resized_video"
# video_path = f"{appdata_folder}/resized_video"
video_path = f"{appdata_folder}/videos"
udf_path = f"{appdata_folder}/oracle/udf"
log_path = f"{appdata_folder}/log"

result_local_path = f"{appdata_folder}/result_img"
result_web_path = f"{appdata_url_path}/result_img"

udf_preview_local_base_path = f"{appdata_folder}/udf_preview"
udf_preview_web_base_path = f"{appdata_url_path}/udf_preview"


def gen_job_id():
    return datetime.now().strftime("%Y%m%d%H%M%S")


class Job(object):
    def __init__(self, process:subprocess.Popen, query):
        self.process = process
        self.query = query

class JobManager(object):

    def __init__(self):
        self.job_dict = {}

    def add_job(self, job_id, process: subprocess.Popen, query):
        self.job_dict[job_id] = Job(process, query)

    def get_if_finished(self, job_id):
        job_obj = self.job_dict[job_id]
        popen_obj = job_obj.process

        if popen_obj.poll() is None:
            return None
        else:
            return job_obj





jobManager = JobManager()

def create_500_error_response(err_msg):
    return {
        "status_code": 500,
        "err_msg": err_msg
    }

# done
@app.route("/api/video_list")
def get_video_list():
    videos = os.listdir(video_path)
    videos = list(filter(lambda x: "." != x[0], videos))

    return jsonify(videos)

# done
@app.route("/api/video_params/<video_name>")
def get_video_params(video_name):
    result = subprocess.run(f"ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,avg_frame_rate,height,width -of default=nokey=1:noprint_wrappers=1 {os.path.join(video_path, video_name)}", stdout=subprocess.PIPE, shell=True).stdout
    entries = result.decode().split("\n")
    response = {}
    response["height"] = entries[0]
    response["width"] = entries[1]
    fps_e, fps_d = entries[2].split("/")
    response["fps"] = float(fps_e) / float(fps_d) 
    response["length"] = entries[3]
    return jsonify(response)

# done but slow
@app.route("/api/video", methods=["POST"])
def upload_video():
    file = request.files['file']
    print(file)
    if file:
        file.save(os.path.join(video_path, file.filename))
    return jsonify({})

# done
@app.route("/api/udf_list")
def get_udf_list():
    udf_files = os.listdir(udf_path)
    udfs = [udf for udf in udf_files if os.path.isdir(os.path.join(udf_path, udf)) and udf != "__pycache__"]
    return jsonify(udfs)

# done
@app.route("/api/udf_params/<udf_name>/params")
def get_udf_params(udf_name):
    udf_class = get_udf_class(udf_name)
    udf = udf_class()
    arg_parser = udf.get_arg_parser()
    response = {}
    for action in arg_parser._actions:
        if action.dest != "help":
            response[action.dest] = action.default
    return jsonify(response)


@app.route("/api/udf_list/<udf_name>")
def get_udf(udf_name):

    udf_file_path = os.path.join(udf_path, udf_name, "score_func.py")
    if not os.path.exists(udf_file_path):
        return jsonify(create_500_error_response(f"Unknow udf path: {udf_file_path}"))

    with open(udf_file_path) as f:
        content = f.read()
        return jsonify({
          "udf_content": content
        })


@app.route("/api/udf_list/<udf_name>", methods=["POST"])
def update_create_udf(udf_name):
    query = request.get_json()
    udf_content = query["udf_content"]

    udf_base_path = os.path.join(udf_path, udf_name)
    udf_file_path = os.path.join(udf_base_path, "score_func.py")

    if not os.path.exists(udf_base_path):
        os.makedirs(udf_base_path)

    with open(udf_file_path, "w") as f:
        f.write(udf_content)

    return jsonify({})


cur_udf_name = ""
cur_udf_params = {}
cur_udf_instance = None
cur_vr = None
cur_video_name = ""


def _process_everest_output(output, udf_name, video_name):
    #match = re.compile("Top-[0-9]* indices: \[(.*)\]")
    match = re.compile("Top-[0-9]* everest frames: \[(.*)\]")
    result_str = match.search(output)
    if result_str is None:
        return None
    print(os.path.dirname(__file__))
    result_str = result_str.group(1)
    topk_idx = [int(i) for i in result_str.split(' ')]

    udf_instance = get_udf_class(udf_name)()
    opt = udf_instance.get_arg_parser().parse_args([])
    udf_instance.initialize(opt)

    vr = DecordVideoReader(os.path.join(video_path, video_name),
                           udf_instance.get_img_size(), is_torch=False)

    imgs = vr.get_batch(topk_idx)
    scores = []
    visual_imgs = []
    batch_size = 16
    for i in range(math.ceil(len(imgs) / batch_size)):
        batch = imgs[i * batch_size:(i + 1) * batch_size]
        if udf_name == "tailgating_danger":
            s = udf_instance.get_scores(batch, False)
            v = [Image.fromarray(m) for m in batch]
        else:
            s, v = udf_instance.get_scores(batch, True)
        scores.extend(s)
        visual_imgs.extend(v)
    return topk_idx, scores, visual_imgs

def get_log_file(query):
    video = query["video"]
    k = query["k"]
    window = query["window"]
    log_file = os.path.join(log_path, f"{os.path.splitext(video)[0]}_{k}_{window}")
    return log_file

@app.route("/api/run/<job_id>")
def get_result(job_id):

    # return jsonify({
    #     "ready": True,
        # mock videos
        # "results":[
        #     {
        #     "img":"/resized_video/archie.mp4"
        # },
        #     {
        #     "img":"/resized_video/archie.mp4"
        # },
        #     {
        #     "img":"/resized_video/archie.mp4"
        # },
        # {
        #     "img":"/resized_video/archie.mp4"
        # },
        # ]
        # mock image
    #     "results":[
    #         {
    #         "img":"/udf_preview/0ba6d569-036a-421d-bac8-603137afe4f8.png"
    #     },
    #         {
    #         "img":"/udf_preview/0ba6d569-036a-421d-bac8-603137afe4f8.png"
    #     },
    #         {
    #         "img":"/udf_preview/0ba6d569-036a-421d-bac8-603137afe4f8.png"
    #     },
    #     {
    #         "img":"/udf_preview/0ba6d569-036a-421d-bac8-603137afe4f8.png"
    #     },
    #     ]
    # })

    job_obj:Job = jobManager.get_if_finished(job_id)

    if job_obj is None:
        return jsonify({"ready": False})

    query = job_obj.query
    udf_name = query["udf"]

    result = job_obj.process
    output = result.stdout.read().decode()
    log_file = get_log_file(query)
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(output)

    topk_idx, scores, visual_imgs = _process_everest_output(output, udf_name, query["video"])
    response = {}
    if result.returncode == 0:
        response["ready"] = True
        response["success"] = True
        response["results"] = []

        img_base_path = os.path.join(result_local_path, job_id)
        os.makedirs(img_base_path, exist_ok=True)

        for f, s, img in zip(topk_idx, scores, visual_imgs):
            if query["window"] == 1:
                img_path = os.path.join(img_base_path,
                                        f"{query['video']}_{f}.png")
                img_web_path = f"{result_web_path}/{job_id}/{query['video']}_{f}.png"
                img.save(img_path)
                timestamp = str(timedelta(seconds=f / 30.0))
            else:
                img_path = os.path.join(img_base_path,
                                        f"{query['video']}_{f}.mp4")
                window = query["window"]
                start_ts = timedelta(seconds=f * window / 30.0)
                end_ts = timedelta(seconds=window / 30.0)
                cmd = f"ffmpeg -y -ss {start_ts} -i {os.path.join(video_path, query['video'])} -to {end_ts} -c copy {img_path}"
                logging.info(cmd)
                subprocess.run(cmd, shell=True)
                img_web_path = f"{result_web_path}/{job_id}/{query['video']}_{f}.mp4"
                timestamp = str(start_ts)
            print(timestamp)
            timestamp = timestamp.split(".")
            if len(timestamp) > 1:
                timestamp = timestamp[0] + "." + timestamp[1][:2]
            else:
                timestamp = timestamp[0]
                
            response["results"].append(
                {"frame": f, "score": s, "img": img_web_path, "timestamp": timestamp})
    else:
        response["success"] = False
    return jsonify(response)









# pass the query parameters in the body of the request (in the form of JSON), but currently since UDF is hardcoded, only the "video, k, thres, window" parameters are used
@app.route("/api/run", methods=['POST'])
def run_everest():

    # return jsonify({
    #     "job_id":1
    # })
    # the config is hardcoded according to the video, todo: generate a config file according to the query params
    query = request.get_json()
    logging.info(f"Run query: {query}")
    # config_map = {
    #    "traffic_footage.mp4": {"config_file": "config/archie_5h.data",
    #                   "udf_name": "obj_counting"},
    #    "dashcam.mov": {"config_file": "", 
    #                   "udf_name": "tailgating_danger"}
    #}
    # config_file = config_map[query["video"]]["config_file"]
    # udf_name = config_map[query["video"]]["udf_name"]
    
    # write config file
    config_str = [f"--video\n{os.path.join('videos' ,query['video'])}\n",f"--k\n{query['k']}\n","--diff_thres\n0.001\n","--num_train\n0.05\n","--num_valid\n0.05\n","--max_score\n1000\n",f"--udf\n{query['udf']}\n","--class_thres\n0.5\n","--save"]
    
    config_path = f"config/{query['video'].split('.')[0]}_{query['k']}.arg"
    
    with open(os.path.join(appdata_folder,config_path),"w") as config_file:
        config_file.writelines(config_str)
    
    log_file = get_log_file(query)
    logging.info(f"everest log file : {log_file}")
    if os.path.exists(log_file):
        cmd = f"cat {log_file}"
    else:
        #cmd=f"cd .. && python3 experiment.py --data_config {config_file} --k {query['k']} --conf {query['thres']} --window {query['window']}"
        cmd = f"cd .. && python3 everest.py @{config_path}"
    job_id = gen_job_id()
    logging.info(f"Run subprocess: {cmd} with job id: {job_id}")

    popen_obj = subprocess.Popen(
        cmd,
        shell=True, stdout=subprocess.PIPE)

    jobManager.add_job(job_id, popen_obj, query)
    response = {"job_id": job_id}

    return jsonify(response)


@app.route("/api/test_udf/<udf_name>/<video_name>/<int:frame>", methods=["POST"])
def test_udf(udf_name, video_name, frame):
    global cur_udf_name
    global cur_udf_params
    global cur_udf_instance

    args = request.get_json()
    udf_param_changed = False
    if cur_udf_name != udf_name:
        cur_udf_instance = get_udf_class(udf_name)()
        opt = dict_to_opt(args, cur_udf_instance.get_arg_parser())
        cur_udf_instance.initialize(opt)
    else:
        try:
            for key,value in cur_udf_params.items():
                if cur_udf_params[key] != args[key]:
                    udf_param_changed = True
                    break
        except KeyError:
            udf_param_changed = True
    
        if udf_param_changed:
            cur_udf_instance.set_args(args)
    
    if cur_video_name != video_name:
        cur_vr = DecordVideoReader(os.path.join(video_path, video_name), cur_udf_instance.get_img_size(), is_torch=False)
    
    img = cur_vr[frame]
    score, visual_img = cur_udf_instance.get_scores(np.array([img]), True)
    score = score[0]
    visual_img = visual_img[0]

    tmp_name = str(uuid.uuid4()) + ".png"
    file_path = os.path.join(udf_preview_local_base_path, tmp_name)
    web_path = f"{udf_preview_web_base_path}/{tmp_name}"

    os.makedirs(udf_preview_local_base_path, exist_ok=True)

    visual_img.save(file_path)
    response = {
        "score": score,
        "img": web_path
    }
    logging.info(f"Respond UDF preview: {response}")
    return jsonify(response)

