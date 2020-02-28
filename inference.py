from utils.parse_config import *
import argparse
import math
import os
import time
import numpy as np
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=30, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/virtualroad.data", help="path to data config file")
    parser.add_argument("--n_cpu", type=int, default=20, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)
    
    prev = time.time()
    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    split_path = data_config["split"]
    distribution_path = data_config["distribution"]
    test_path = split_path + "/test.txt"
    test_idx = np.load(test_path.replace(".txt", ".npy"))


    with open(test_path, "rt") as f:
        path_list = f.readlines()
    
    num = len(path_list)
    print(num)
    subtask_len = 3600 * 5 * 30
    print("num of subtasks:", math.ceil(num / subtask_len))
    for i in range(math.ceil(num / subtask_len)):
        subtask_path = split_path + "/test_%d.txt" % i
        with open(subtask_path, "wt") as f:
            for p in path_list[i*subtask_len: (i+1)*subtask_len]:
                f.write(p)
        np.save(split_path + "/test_%d.npy" % i, test_idx[i*subtask_len: (i+1)*subtask_len])
        if os.system("python inference_subtask.py --data_config %s --batch_size %d --n_cpu %d --img_size %d --subtask %d" % (opt.data_config, opt.batch_size, opt.n_cpu, opt.img_size, i)) != 0:
            print("subtask %d failed" % i)
            sys.exit(1)

    discarded_list = []
    remained_list = []
    pi_list = []
    sigma_list = []
    mu_list = []
    for i in range(math.ceil(num / subtask_len)):
        pi_list.append(np.load(distribution_path + "/pi_%d.npy" % i))
        mu_list.append(np.load(distribution_path + "/mu_%d.npy" % i))
        sigma_list.append(np.load(distribution_path + "/sigma_%d.npy" % i))
        discarded_list.append(np.load(split_path + "/discarded_%d.npy" % i))
        remained_list.append(np.load(split_path + "/remained_%d.npy" %i))

    discarded = np.concatenate(discarded_list, 0)
    remained = np.concatenate(remained_list, 0)
    total_pi = np.concatenate(pi_list, 0)
    total_sigma = np.concatenate(sigma_list, 0)
    total_mu = np.concatenate(mu_list, 0)
    print("Time:", time.time() - prev)
    print("Discarded %d/%d frames" % (len(discarded), num))
    np.save(distribution_path + "/mu.npy", total_mu)
    np.save(distribution_path + "/sigma.npy", total_sigma)
    np.save(distribution_path + "/pi.npy", total_pi)
    np.save(split_path + "/discarded.npy", discarded)
    np.save(split_path + "/remained.npy", remained)
