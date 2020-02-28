from utils.parse_config import *

import os
import sys
import time
import argparse

import numpy as np
import random

import torch
from config import *
from torch.distributions import normal
from topk import groundtruth_shortcut
from utils.topk_utils import *
import math

def load_distribution(data_config, window):

    distribution_path = data_config["distribution"]
    pi = np.load(distribution_path + "/pi.npy")
    mu = np.load(distribution_path + "/mu.npy")
    sigma = np.load(distribution_path + "/sigma.npy")
    if window < 2:
        return pi, mu, sigma

    obj = int(data_config["object"])
    split_path = data_config["split"]
    all_path = split_path + "/all.txt"

    with open(all_path, "rt") as f:
        path_list = f.readlines()
        path_list = [p.rstrip() for p in path_list]

    num_mixtures = pi.shape[1]
    train_idx = np.load(split_path + "/train.npy")
    remained_ref = np.load(split_path + "/remained.npy")
    remained_idx = remained_ref[:,0]
    discarded_ref = np.load(split_path + "/discarded.npy")
    val_idx = np.load(split_path + "/val.npy")

    train_scores = np.array(groundtruth_shortcut(idx2path(path_list, train_idx), obj))
    val_scores = np.array(groundtruth_shortcut(idx2path(path_list, val_idx), obj))
    
    num = len(path_list)
    pi_all = np.zeros([num, num_mixtures], dtype=np.float32)
    sigma_all = np.zeros([num, num_mixtures], dtype=np.float32)
    mu_all = np.zeros([num, num_mixtures], dtype=np.float32)

    pi_all[train_idx, 0] = 1.0
    mu_all[train_idx, 0] = train_scores
    sigma_all[train_idx, :] = 0.1
    pi_all[val_idx, 0] = 1.0
    mu_all[val_idx, 0] = val_scores
    sigma_all[val_idx, :] = 0.1
    pi_all[remained_idx] = pi
    mu_all[remained_idx] = mu
    sigma_all[remained_idx] = sigma
    pi_all[discarded_ref[:, 0]] = pi_all[discarded_ref[:, 1]]
    mu_all[discarded_ref[:, 0]] = mu_all[discarded_ref[:, 1]]
    sigma_all[discarded_ref[:, 0]] = sigma_all[discarded_ref[:, 1]]

    assert len((sigma_all[:,0] == 0).nonzero()[0]) == 0
    
    pi = pi_all
    mu = mu_all
    sigma = sigma_all
    mu_bar = (pi * mu).sum(-1, keepdims=True)
    sigma_bar = (pi * (sigma**2 + mu**2 - mu_bar**2)).sum(-1)
    
    num_windows = int(num / window)
    reshaped_mu_bar = np.reshape(mu_bar[: num_windows * window], [num_windows, window])
    reshaped_sigma_bar = np.reshape(sigma_bar[: num_windows * window], [num_windows, window])
    mu = reshaped_mu_bar.mean(-1)
    sigma = reshaped_sigma_bar.mean(-1)
    if num_windows * window != num:
        single_mu = mu_bar[num_windows * window:].mean()
        single_sigma = sigma_bar[num_windows * window:].mean()
        single_mu = np.reshape(single_mu, [1])
        single_sigma = np.reshape(single_sigma, [1])
        mu = np.concatenate([mu, single_mu], 0)
        sigma = np.concatenate([sigma, single_sigma], 0)
        num_windows += 1

    sigma = np.sqrt(sigma)
    pi = np.ones([num_windows, 1], dtype=np.float32)
    return pi, np.reshape(mu, [-1, 1]), np.reshape(sigma, [-1, 1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="config/virtualroad.data", help="path to data config file")
    parser.add_argument("--batch_size", type=int, default=1e3)
    parser.add_argument("--window", type=int, default=1, help="window size")
    opt = parser.parse_args()
    opt.batch_size = int(opt.batch_size)
    print(opt)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    distribution_path = data_config["distribution"]
    cdf_path = distribution_path + "/cdf.npy"
    max_score = int(data_config["max_score"])

    pi, mu, sigma = load_distribution(data_config, opt.window)

    num = len(pi)
    print(num)
    opt.batch_size = min(num, opt.batch_size)

    cdf_list = []
    ticks = torch.arange(0.5, max_score + 0.5, 1, device=device).view(1, 1, max_score)
    for b in range(math.ceil(num / opt.batch_size)):
        pi_gpu = torch.from_numpy(pi[b * opt.batch_size: (b+1) * opt.batch_size]).to(device)
        mu_gpu = torch.from_numpy(mu[b * opt.batch_size: (b+1) * opt.batch_size]).to(device)
        sigma_gpu = torch.from_numpy(sigma[b * opt.batch_size: (b+1) * opt.batch_size]).to(device)
        normals = normal.Normal(mu_gpu.unsqueeze(-1), sigma_gpu.unsqueeze(-1))
        cdf = normals.cdf(ticks)
        cdf = (cdf * pi_gpu.unsqueeze(-1)).sum(1)
        cdf = torch.where(cdf >= 0.997, torch.ones(cdf.shape, device=device), cdf)
        cdf = cdf.clamp(1e-20, 1)
        cdf_list.append(cdf.cpu().numpy())
        torch.cuda.empty_cache()
    
    cdf = np.concatenate(cdf_list, 0)
    np.save(cdf_path, cdf) 
       
