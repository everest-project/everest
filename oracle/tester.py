import argparse
import importlib
import re
from utils.video_reader import DecordVideoReader
import os

class UDFTester:
    def __init__(self, udf_name, video_path, gpu=None):
        class_name = re.sub(r'_([a-z])', lambda m: m.group(1).upper(), udf_name)
        udf_class = importlib.import_module(f"oracle.{udf_name}.score_func.{class_name}")
        self.udf = udf_class()
        self.vr = DecordVideoReader(video_path, self.udf.get_img_size(), gpu=gpu, is_torch=False)
    
    def get_udf_default_args(self):
        return
    
    def init_udf(self, udf_args):
        parser = self.udf.get_arg_parser()
        opt, _ = paser.parse_known_args(udf_args)
        self.udf.initialize(opt, gpu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="/mnt/resized_video/archie.mp4")
    parser.add_argument("--udf", type=str, default="obj_counting")
    parser.add_argument("--frames", type=str, default="0,37800")
    parser.add_argument("--output", type=str, default="result")
    opt, _ = parser.parse_known_args()

    class_name = "".join([s.capitalize() for s in opt.udf.split("_")])
    udf_module = importlib.import_module(f"oracle.udf.{opt.udf}.score_func")
    udf_class = getattr(udf_module, class_name)
    udf = udf_class()
    udf_arg_parser = udf.get_arg_parser()
    udf_opt, _ = udf_arg_parser.parse_known_args()
    udf.initialize(udf_opt)
    vr = DecordVideoReader(opt.video, udf.get_img_size(), is_torch=False)

    frames = [int(f) for f in opt.frames.split(",")]
    imgs = vr.get_batch(frames)
    scores, visual_imgs = udf.get_scores(imgs, True)
    print(scores)
    for frame, visual_img in zip(frames, visual_imgs):
        visual_img.save(os.path.join(opt.output, f"{frame}.jpg"))
        
    



