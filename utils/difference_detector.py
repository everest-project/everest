import os
import sys

import random
import argparse
from utils.parse_config import *
from utils.datasets import *

import torch
import tqdm
import time
import random

if __name__ == "__main__":
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="config/coral.data", help="path to data config file")
    parser.add_argument("--window", default=1)
    parser.add_argument("--threshold", default=10)
    opt = parser.parse_args()
    print(opt)

    data_config = parse_data_config(opt.data_config)
    image_path = data_config["images"]
    discarded_path = data_config["discarded"]
    train_path = data_config["train"]
    test_path = data_config["test"]
    valid_path = data_config["valid"]
    all_path = data_config["all"]
    num_train = int(data_config["num_train"])
    num_valid = int(data_config["num_valid"])

    paths = os.listdir(image_path)
    paths = [image_path + "/" + p for p in paths if p.endswith(".jpg")]
    paths.sort(key=lambda x: (len(x), x))

    dataset = ImageLoader(paths)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=dataset.collate_fn
    )

    prev = time.time()
    reference = None
    discarded = []
    remained = []
    for i, (img_path, img) in enumerate(tqdm.tqdm(dataloader, desc="Difference detecting")):
        pass
        #if reference is None or ((img[0] - reference)**2).mean() > opt.threshold or (opt.window >= 30 and i % opt.window==0):
        #    remained.append((i, img_path))
        #    reference = img[0]
        #else:
        #    discarded.append((i, img_path))
    
    random.shuffle(remained)
    train = remained[:num_train]
    valid = remained[num_train:num_train+num_valid]
    test = remained[num_train+num_valid:]

    train.sort()
    valid.sort()
    test.sort()

    print("runtime:", time.time() - prev)

    def write_list(path, l):
        with open(path, "wt") as f:
            for p in l:
                f.write(f, p + "\n")


    write_list(all_path, paths)
    write_list(discarded_path, [t[1] for t in discarded])
    write_list(train_path, [t[1] for t in train])
    write_list(valid_path, [t[1] for t in valid])
    write_list(test_path, [t[1] for t in test])
    
    np.save(discarded_path.replace(".txt", ".npy"), np.array([t[0] for t in discarded]))
    np.save(train_path.replace(".txt", ".npy"), np.array([t[0] for t in train]))
    np.save(valid_path.replace(".txt", ".npy"), np.array([t[0] for t in valid]))
    np.save(test_path.replace(".txt", ".npy"), np.array([t[0] for t in test]))

