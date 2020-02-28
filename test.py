from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from config import *

def evaluate_video(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, n_cpu, obj):
    model.eval()

    # Get dataloader
    dataset = VideoObjectDataset(path, img_size=img_size, augment=False, obj=obj)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu, collate_fn=dataset.collate_fn)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    mdn_tp = 0
    mdn_acc = 0
    mean_dict = {}
    var_dict = {}
    for batch_i, (_, imgs, targets, scores) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.to(device), requires_grad=False)

        with torch.no_grad():
            outputs, mdn_output = model(imgs)
            pi, sigma, mu = mdn_output[2], mdn_output[3], mdn_output[4]
            mean = (mu * pi).sum(-1)
            var = ((sigma**2 + mu**2 - mean.unsqueeze(-1)**2) * pi).sum(-1)
            mdn_tp += (torch.round(mean) == scores).sum().item()
            mdn_acc += len(pi)
            for i in range(len(mean)):
                lab = scores[i].item()
                if lab not in mean_dict:
                    mean_dict[lab] = [mean[i].item()]
                    var_dict[lab] = [var[i].item()]
                else:
                    mean_dict[lab] += [mean[i].item()]
                    var_dict[lab] += [var[i].item()]

    mdn_acc = mdn_tp / mdn_acc
    print('mdn accuracy: {:.2f}'.format(mdn_acc))
    print('K N Mean Var MSE')
    se_dict = dict()
    count_dict = dict()
    keys = [int(k) for k in mean_dict.keys()]
    keys.sort()
    for k in keys:
        samples = len(mean_dict[k])
        se = ((np.array(mean_dict[k]) - k) ** 2).sum()
        mean = np.mean(mean_dict[k])
        var = np.mean(var_dict[k])
        se_dict[k] = se
        count_dict[k] = samples
        print('{}: {} {:.2f} {:.2F} {:.2f}'.format(k, samples, mean, var, se / samples))
    se = np.array(list(se_dict.values()))
    count = np.array(list(count_dict.values()))
    print('MSE: {:.2f}'.format(se.sum() / count.sum()))





def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.to(device), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
