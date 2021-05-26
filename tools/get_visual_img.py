from utils.video_reader import DecordVideoReader
from utils.parse_config import parse_data_config
from utils.label_reader import *
from utils.utils import *
from oracle.utils import *
import torch
import lmdb
import decord
import os
import config
import tqdm
import sys
from os import path as osp
import numpy as np
import config
import io
from PIL import Image
LMDB_MAP_SIZE = 1 << 40

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="videos/archie.mp4")
    parser.add_argument("--udf", type=str, default="number_of_cars")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    opt, _ = parser.parse_known_args()
    print(opt)

    udf = get_udf_class(opt.udf)()
    udf_arg_parser = udf.get_arg_parser()
    udf_opt, _ = udf_arg_parser.parse_known_args()
    udf.initialize(udf_opt, opt.gpu)

    output_path = get_cached_gt_path(opt)
    vr = DecordVideoReader(opt.video, udf.get_img_size(), is_torch=False)
    num_batches = len(vr) // opt.batch
    batches = [list(range(i * opt.batch, (i+1) * opt.batch)) for i in range(num_batches)]

    scores = []
    visual_imgs = []
    for batch in tqdm.tqdm(batches, desc="labeling"):
        if len(batch) > 0:
            imgs = vr.get_batch(batch)
            score,visual_img =udf.get_scores(imgs, visualize=True)

            scores.extend(score)
            visual_imgs.extend(visual_img)
    for visual_img in visual_imgs:
        if visual_img:
            print (visual_img)
            visual_img.save("test.jpg")
            break
    batch =list(range(140,156))

    udf.get_scores(vr.get_batch(batch),visualize=True)[1][0].save("test2.jpg")

   # print(image[4])
    #image[0][0].save("test.jpg")    
    os.makedirs(config.cached_gt_dir, exist_ok=True)
    np.save(output_path, np.array(scores))
