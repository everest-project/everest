import os
import sys

import random
import argparse
from utils.parse_config import *
import numpy as np

if __name__ == "__main__":
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="config/virtualroad.data", help="path to data config file")
    opt = parser.parse_args()
    print(opt)

    data_config = parse_data_config(opt.data_config)
    image_path = data_config["images"]
    split_path = data_config["split"]
    train_path = split_path + "/train.txt"
    test_path = split_path + "/test.txt"
    valid_path = split_path + "/val.txt"
    all_path = split_path + "/all.txt"
    num_train = int(data_config["num_train"])
    num_valid = int(data_config["num_valid"])

    images = os.listdir(image_path)
    images.sort(key=lambda x: (len(x), x))
    images = [im for im in images if im.endswith(".jpg")]
    images.sort(key=lambda x: (len(x), x))
    images = [i for i in enumerate(images)]

    random.shuffle(images)
    train = images[:num_train]
    valid = images[num_train:num_train+num_valid]
    test = images[num_train+num_valid:]

    train.sort()
    test.sort()
    valid.sort()

    def write_list(path, l):
        with open(path, "wt") as f:
            for p in l:
                f.write(image_path + "/" + p + "\n")

    write_list(all_path, [t[1] for t in images])
    write_list(train_path, [t[1] for t in train])
    write_list(valid_path, [t[1] for t in valid])
    write_list(test_path, [t[1] for t in test])

    np.save(train_path.replace(".txt", ".npy"), np.array([t[0] for t in train]))
    np.save(valid_path.replace(".txt", ".npy"), np.array([t[0] for t in valid]))
    np.save(test_path.replace(".txt", ".npy"), np.array([t[0] for t in test]))

