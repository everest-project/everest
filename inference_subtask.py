from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate, evaluate_video

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
from config import *
import tqdm
import time
import resource

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=30, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/virtualroad.data", help="path to data config file")
    parser.add_argument("--n_cpu", type=int, default=20, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--subtask", type=int, default=0, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)
    print("Runing subtask", opt.subtask)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    split_path = data_config["split"]
    model_def = data_config["model_def"]
    test_path = split_path + "/test_%d.txt" % opt.subtask
    test_idx = test_path.replace(".txt", ".npy")
    weight_path = data_config["backup"]
    diff_thres = float(data_config["diff_thres"])
    distribution_path = data_config["distribution"]
    obj = int(data_config["object"])

    # Initiate model
    model = Darknet(model_def).to(device)
    model.load_state_dict(torch.load(weight_path))

    # Get dataloader
    dataset = VideoObjectDataset_Diff(test_path, test_idx, img_size=opt.img_size, threshold=diff_thres, obj=obj, ref_dist=opt.batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        pin_memory=False,
        collate_fn=dataset.collate_fn
    )
    model.eval()

    total_pi_list = []
    total_sigma_list = []
    total_mu_list = []
    discarded_list = []
    remained_list = []

    with torch.no_grad():
        for batch_i, (imgs, discarded, remained) in enumerate(tqdm.tqdm(dataloader, desc="Inferencing")):
            imgs = imgs.to(device)
            _, mdn_output = model(imgs)
            pi, sigma, mu = mdn_output[2], mdn_output[3], mdn_output[4]
            total_pi_list.append(pi)
            total_sigma_list.append(sigma)
            total_mu_list.append(mu)
            discarded_list.append(discarded)
            remained_list.append(remained)

    discarded = torch.cat(discarded_list, 0)
    remained = torch.cat(remained_list, 0)
    total_pi = torch.cat(total_pi_list, 0)
    total_sigma = torch.cat(total_sigma_list, 0)
    total_mu = torch.cat(total_mu_list, 0)
    np.save(distribution_path + "/mu_%d.npy" % opt.subtask, total_mu)
    np.save(distribution_path + "/sigma_%d.npy" % opt.subtask, total_sigma)
    np.save(distribution_path + "/pi_%d.npy" % opt.subtask, total_pi)
    np.save(split_path + "/discarded_%d.npy" % opt.subtask, discarded)
    np.save(split_path + "/remained_%d.npy" % opt.subtask, remained)
