from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate, evaluate_video

from terminaltables import AsciiTable

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

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=1, help="number of gradient accums before step")
    parser.add_argument("--data_config", type=str, default="config/virtualroad.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="weights/yolov3-tiny.weights", help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    model_def = data_config["model_def"]
    split_path = data_config["split"]
    train_path = split_path + "/train.txt"
    valid_path = split_path + "/val.txt"
    obj = int(data_config["object"])
    class_names = ["car"]

    # Initiate model
    model = Darknet(model_def).to(device)
    model.apply(weights_init_normal)

    #If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = VideoObjectDataset(train_path, augment=True, img_size=opt.img_size, obj=obj)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    prev = time.time()
    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets, scores) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            scores = Variable(scores.to(device), requires_grad=False)
            
            yolo_loss, outputs, mdn_output = model(imgs, targets, scores, epoch=epoch)
            wta_loss = mdn_output[0]
            mdn_loss = mdn_output[1]

            #loss = 0.1 * yolo_loss + 0.1 * wta_loss + 1 * mdn_loss
            if epoch < opt.epochs / 10 * 4:
                loss = yolo_loss + 0.9 * wta_loss + 0.0 * mdn_loss
            else:
                loss = 0.1 * yolo_loss + 0.5 * wta_loss + 1 * mdn_loss
            loss.backward()

            if batches_done % opt.gradient_accumulations == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

            # mdn metric
            mdn_loss = to_cpu(mdn_loss)
            pi, sigma, mu, h = mdn_output[2], mdn_output[3], mdn_output[4], mdn_output[5]
            print(pi[0])
            print(mu[0])
            print(sigma[0])
            print(h[0])
            mean = (pi * mu).sum(-1)
            print(mean)
            var = ((sigma**2 + mu**2 - mean.unsqueeze(-1)**2) * pi).sum(-1)
            var = var.mean()
            mdn_acc = (torch.round(mean) == to_cpu(scores)).sum() / float(len(pi))

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            log_str += "\nmdn_loss: %.2f, wta_loss: %.2f mdn_acc: %.3f mean: %.3f var: %.3f" % (wta_loss.item(), mdn_loss.item(), mdn_acc.item(), mean.mean().item(), var.item())

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

    print("runtime:", time.time()-prev)

    print("\n---- Evaluating Model ----")
   # Evaluate the model on the validation set
    evaluate_video(
        model,
        path=valid_path,
        iou_thres=0.5,
        conf_thres=0.5,
        nms_thres=0.5,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        n_cpu=opt.n_cpu,
        obj=obj
        )

    torch.save(model.state_dict(), data_config["backup"])

