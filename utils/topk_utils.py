import os
import json
import decord
import torch
import math
import numpy as np
from tqdm import tqdm
from utils.utils import non_max_suppression
from models import Darknet
from decord import VideoReader, gpu

def f2idx(idx_list, fs):
    return [idx_list[f] for f in fs]

def idx2path(path_list, idxs):
    return [path_list[i] for i in idxs]

def precision(approx, exact):
    k = len(approx)
    exact = exact[:k]
    tp = 0
    exact_dict = dict()
    for v in exact:
        if v in exact_dict:
            exact_dict[v] += 1
        else:
            exact_dict[v] = 1
    for v in approx:
        if v in exact_dict and exact_dict[v] > 0:
            tp += 1
            exact_dict[v] -= 1
    return tp / k

def rank_distance(approx, exact):
    print(approx)
    k = len(approx)
    total_dist = 0
    for i in range(k):
        # take tie into account 
        if i == 0:
            rank = 0
        elif approx[i] != approx[i-1]:
            rank = i
        j = np.searchsorted(exact[::-1], approx[i], side='right')
        real_rank = len(exact) - j
        total_dist += abs(real_rank - rank)
    return total_dist / k

def score_error(approx, exact):
    k = len(approx)
    approx_ = np.array(approx)
    exact_ = np.array(exact)[:k]
    err = np.sum(np.abs(approx_ - exact_))
    return err / k

def evaluate(topk, k, label_reader, window=1, data_size=None, cached_gt_path=None):
    topk_scores = [sf.s for sf in topk]
    topk_idxs = [sf.f for sf in topk]
    if data_size is None:
        data_size = len(label_reader)
    if cached_gt_path is not None and os.path.exists(cached_gt_path):
        cached_gt = np.load(cached_gt_path)
    else:
        cached_gt = np.array(label_reader.get_batch(range(data_size)))
    if window < 2:
        gts = cached_gt
    else:
        num_frames = data_size   
        num_windows = math.ceil(num_frames / window)
        scores = []
        for w in range(num_windows):
            score = cached_gt[list(range(w * window, min(num_frames-1, (w+1) * window)))]
            score = int(round(np.mean(score)))
            scores.append(score)
        gts = np.array(scores)
    gt_idxs_all = gts.argsort()[::-1]
    gt_scores = gts[gt_idxs_all]
    gt_idxs = gt_idxs_all[:k]
    prec = precision(topk_scores, gt_scores)
    rd = rank_distance(topk_scores, gt_scores)
    se = score_error(topk_scores, gt_scores)

    print('----------------------------------------')
    print('Top-{} everest scores: {}'.format(k, np.array(topk_scores)))
    print('Top-{} oracle  scores: {}'.format(k, gt_scores[:k]))
    print('Top-{} everest frames: {}'.format(k, np.array(topk_idxs)))
    print('Top-{} oracle  frames: {}'.format(k, gt_idxs))
    print('----------------------------------------')
    return prec, rd, se