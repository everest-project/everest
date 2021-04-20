import os
import sys
import time
import random
import argparse
import numpy as np

from utils.video_reader import *
import config
from utils.topk_utils import *
from utils.label_reader import *
from oracle.utils import *
from phase1 import *
from phase2 import topk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--video", type=str, default="videos/archie.mp4", help="path to the video of interest")
    parser.add_argument("--length", type=int, default=None, help="specify the length of the video, full length by default")
    parser.add_argument("--diff_thres", type=float, help="threshold of the difference detector")
    parser.add_argument("--num_train", type=float, default=0.005, help="training set size of the CMDN")
    parser.add_argument("--num_valid", type=float, default=3000, help="validation set size of the CMDN")
    parser.add_argument("--max_score", type=int, default=50, help="the maximum score")
    parser.add_argument("--udf", type=str, default="number_of_cars", help="the name of the scoring UDF")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--cmdn_train_epochs", type=int, default=10)
    parser.add_argument("--cmdn_train_batch", type=int, default=64)
    parser.add_argument("--cmdn_scan_batch", type=int, default=60)
    parser.add_argument("--oracle_batch", type=int, default=8)
    parser.add_argument("--conf_thres", type=float, default=0.9)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--window_samples", type=float, default=0.1)
    parser.add_argument("--udf_batch", type=int, default=8)
    parser.add_argument("--skip_train_cmdn", default=False, action="store_true")
    parser.add_argument("--skip_cmdn_scan", default=False, action="store_true")
    parser.add_argument("--skip_topk", default=False, action="store_true")
    parser.add_argument("--save", default=False, help="save intermediate results", action="store_true")
    opt, _ = parser.parse_known_args()

    if opt.gpu is not None:
        config.device = torch.device("cuda:%d" % opt.gpu)
        config.decord_ctx = decord.gpu(opt.gpu)

    cached_gt_path = get_cached_gt_path(opt)
    split_path = get_split_path(opt)
    checkpoint_dir = get_checkpoint_dir(opt)

    udf = get_udf_class(opt.udf)()
    vr = DecordVideoReader(opt.video, img_size=config.cmdn_input_size, offset=opt.offset)
    lr = CachedGTLabelReader(cached_gt_path, opt.offset)
    
    if opt.skip_train_cmdn:
        train_idx = np.load(os.path.join(split_path, "train_idxs.npy"))
        valid_idx = np.load(os.path.join(split_path, "valid_idxs.npy"))
        test_idx = np.load(os.path.join(split_path, "test_idxs.npy"))
        for fname in os.listdir(checkpoint_dir):
            if "best" in fname:
                best_model_path = os.path.join(checkpoint_dir, fname)
                break
    else:
        train_idx, valid_idx, test_idx, weight = split_dataset(opt, vr, lr, opt.save)
        best_model_path = train_cmdn(opt, vr, lr, train_idx, valid_idx, weight)
    
    if opt.skip_cmdn_scan:
        remained_ref = np.load(os.path.join(split_path, "remained.npy"))
        discarded_ref = np.load(os.path.join(split_path, "discarded.npy"))
        pi = np.load(os.path.join(split_path, "pi.npy"))
        mu = np.load(os.path.join(split_path, "mu.npy"))
        sigma = np.load(os.path.join(split_path, "sigma.npy"))
    else:
        mu, sigma, pi, discarded_ref, remained_ref = cmdn_scan(opt, best_model_path, vr, test_idx, opt.save)
    
    if opt.window > 1:
        pi, mu, sigma = window_distribution(opt, train_idx, valid_idx, mu, sigma, pi, discarded_ref, remained_ref, lr, vr)
    
    if not opt.skip_topk:
        cdf = gen_cdf(pi, mu, sigma, opt.max_score)
        topk(opt, cdf, remained_ref, train_idx, valid_idx, lr, vr)

