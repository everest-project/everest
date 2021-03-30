import copy
import heapq
import json
import logging
import numpy as np
import torch
from collections import namedtuple
import os
import sys
import time
from scipy.special import logsumexp
from utils.parse_config import *
import config
from blist import sortedlist
from utils.topk_utils import *
import argparse
import random
import math
from utils.video_reader import *
from utils.label_reader import *
random.seed(0)

SF = namedtuple('SF', ['s', 'f'])
class SelectionHeap():
    def __init__(self, batch_size, uncertain_table, certain_table):
        self.ut = uncertain_table
        self.ct = certain_table
        self.size = len(uncertain_table.log_cdf)
        self.batch_size = batch_size
        if certain_table.lam == 0:
            self.LF = np.zeros([self.size])
        else:
            self.LF = self.compute_all_L()
        self.order = np.argsort(self.LF)[::-1]
        self.certain_idxs = []

    def compute_all_L(self):
        log_cdf = self.ut.log_cdf
        return np.log(1 - np.exp(log_cdf[:,self.ct.lam])) - log_cdf[:,self.ct.mu]

    def compute_L(self, f):
        log_cdf = self.ut.log_cdf[f]
        return np.log(1 - np.exp(log_cdf[self.ct.lam])) - log_cdf[self.ct.mu]

    def compute_E(self, f, topk_prob):
        lam = self.ct.lam
        mu = self.ct.mu
        log_cdf = self.ut.log_cdf[f]
        log_pdf = self.ut.log_pdf[f]
        terms = np.zeros([2 + mu-lam])
        terms[0] = topk_prob
        terms[-1] = self.ut.cum(mu) - log_cdf[mu] + np.log(1 - np.exp(log_cdf[mu]))
        mid = np.arange(lam+1, mu+1)
        terms[1:-1] = self.ut.cum(mid) - log_cdf[mid] + log_pdf[mid]
        E = logsumexp(terms)
        if math.isnan(E):
            E = 0
        return E

    def update_order(self):
        self.LF = self.compute_all_L()
        self.order = np.argsort(self.LF)[::-1]
        self.certain_idxs = []

    def bootstrap(self, size):
        return np.argsort(self.ut.log_cdf[:,0])[:size]

    def select(self, topk_prob):
        candidate_heap = PQueue(self.order.shape[0])
        gamma = self.ut.cum(self.ct.mu)
        idx = 0
        while idx < self.order.shape[0]:
            f = self.order[idx]
            if self.ct.is_certain(f):
                self.certain_idxs.append(idx)
                idx += 1
                continue
            E = self.compute_E(f, topk_prob)
            candidate_heap.push(SF(E, f))
            U = logsumexp([topk_prob, self.LF[self.order[idx]] + gamma])
            if math.isnan(U):
                U = 0
            if candidate_heap.size >= self.batch_size and candidate_heap.max(1).s >= U:
                break
            idx += 1
        if len(self.certain_idxs) > 1000:
            self.order = np.delete(self.order, self.certain_idxs)
            self.certain_idxs = []
        return candidate_heap.top(min(self.order.shape[0], self.batch_size))

class UncertainTable():
    def __init__(self, cdf, certain_table):
        self.ct = certain_table
        self.log_cdf = np.log(cdf)
        self.H = self.log_cdf.sum(0)
        self.log_pdf = np.zeros(cdf.shape)
        self.log_pdf[:,0] = cdf[:,0]
        self.log_pdf[:,1:] = cdf[:,1:] - cdf[:,:-1]
        self.log_pdf = np.log(self.log_pdf)

    def cum(self, s):
        return self.H[s] - self.ct.H[s]

    def topk_prob(self, lam):
        return self.cum(lam)

class CertainTable():
    def __init__(self, k, cdf, idx_list=None, initial=None, initial_scores=None):
        initial_sfs = []
        if initial is not None:
            initial_sfs = [SF(s, f) for (s,f) in zip(initial_scores, initial)]
            initial_sfs.sort(reverse=True)
        self.topk = sortedlist(initial_sfs[:k])
        self.lam = 0 if len(self.topk) == 0 else self.topk[0].s
        self.mu = self.lam if len(self.topk) <= 1 else self.topk[1].s
        self.H = np.zeros([cdf.shape[1]])
        self.bitmap = np.zeros([cdf.shape[0]], dtype='bool')
        self.k = k
        self.idx_list = idx_list

    def insert_sw(self, sw, log_cdf):
        self.insert_mirror(sw)
        self.H += log_cdf
        self.bitmap[sw.f] = True

    def insert_sf(self, sf, log_cdf):
        self.insert_mirror(SF(sf.s, f2idx(self.idx_list, [sf.f])[0]))
        self.H += log_cdf
        self.bitmap[sf.f] = True

    def insert_mirror(self, s_idx):
        self.topk.add(s_idx)
        if len(self.topk) > self.k:
            self.topk.pop(0)
        self.lam = 0 if len(self.topk) == 0 else self.topk[0].s
        self.mu = self.lam if len(self.topk) <= 1 else self.topk[1].s

    def is_certain(self, f):
        return self.bitmap[f]

    def topk_mean(self):
        return np.array([sf.s for sf in self.topk]).mean()

class PQueue(object):
    def __init__(self, capacity=0, initializer_list=None):
        self.capacity_ = capacity
        if initializer_list is not None:
            initializer_list = [SF(-s, f) for (s, f) in initializer_list]
        self.size_ = 0 if initializer_list is None else len(initializer_list)
        self.pqueue_ = [] if initializer_list is None else initializer_list

        if self.capacity_ != 0 and self.size_ > self.capacity_:
            self.size_ = self.capacity_
            self.pqueue_ = sorted(self.pqueue_)[:self.size_]

        heapq.heapify(self.pqueue_)

    def __len__(self):
        return self.size_

    def __str__(self):
        return 'PQueue({0})'.format(self.pqueue_.__str__())

    @property
    def size(self):
        return self.size_

    @property
    def capacity(self):
        return self.capacity_

    @property
    def data(self):
        return self.pqueue_

    def top(self, k):
        raw = heapq.nsmallest(self.size_, self.pqueue_)[:k]
        return [SF(-s, f) for (s, f) in raw]

    def push(self, value):
        self.size_ += 1
        value = SF(-value.s, value.f)
        heapq.heappush(self.pqueue_, value)

    def pop(self):
        value = None
        if self.size_ > 0:
            value = heapq.heappop(self.pqueue_)
            self.size_ -= 1
        else:
            raise IndexError

        return SF(-value.s, value.f)

    def min(self, n=1):
        value = None
        if self.size_ >= n:
            value = heapq.nlargest(n, self.pqueue_)[-1]
        else:
            raise IndexError

        return SF(-value.s, value.f)

    def max(self, n=1):
        value = None
        if self.size_ >= n:
            value = heapq.nsmallest(n, self.pqueue_)[-1]
        else:
            raise IndexError

        return SF(-value.s, value.f)

    def mean(self):
        return np.mean([-x[0] for x in self.pqueue_])


def infer_window_gt(ws, window_size, window_samples, label_reader, data_size=None):
    scores = []
    if data_size is None:
        data_size = len(label_reader)
    for w in ws:
        idxs = list(range(w * window_size, min(data_size, (w+1)*window_size)))
        random.shuffle(idxs)
        window_scores = label_reader.get_batch(idxs[:window_samples])
        scores.append(int(round(np.mean(window_scores))))
    return scores

def infer_frame_gt(indices, label_reader):
    return label_reader.get_batch(indices)

def topk(opt, cdf, remained_ref, train_idx, val_idx, lr, vr):
    k = opt.k
    conf_thres = opt.conf_thres
    window = opt.window
    window_samples = int(opt.window * opt.window_samples)
    batch_size = opt.udf_batch
    max_score = opt.max_score
    data_size = get_video_length(opt, vr)

    if window < 2:
        initial_certain = np.concatenate([train_idx, val_idx], 0)
        initial_certain_scores = infer_frame_gt(initial_certain, lr)
        idx_list = remained_ref[:,0]
        certain_table = CertainTable(k, cdf, idx_list, initial_certain, initial_certain_scores)
    else:
        certain_table = CertainTable(k, cdf)

    uncertain_table = UncertainTable(cdf, certain_table)
    select_heap = SelectionHeap(batch_size, uncertain_table, certain_table)

    niter = 0
    prev_lam, prev_mu = -1, -1
    candidates = []

    while select_heap.size > 0:
        topk_prob = uncertain_table.topk_prob(certain_table.lam)
        if topk_prob >= np.log(conf_thres):
            break

        prev = time.time()
        if len(certain_table.topk) < k:
            print("bootstraping")
            clean_f = select_heap.bootstrap(k - len(certain_table.topk))
        else:
            candidates = select_heap.select(topk_prob)
            clean_f = [sf.f for sf in candidates]

        if window < 2:
            scores = infer_frame_gt(f2idx(idx_list, clean_f), lr)
            for f, score in zip(clean_f, scores):
                certain_table.insert_sf(SF(score, f), uncertain_table.log_cdf[f])
                mirrors = remained_ref[f, 2:]
                for m in mirrors:
                    if m == -1:
                        break
                    certain_table.insert_mirror(SF(score, m))
        else:
            scores = infer_window_gt(clean_f, window, window_samples, lr, data_size)
            for w, score in zip(clean_f, scores):
                certain_table.insert_sw(SF(score, w), uncertain_table.log_cdf[w])
        select_heap.size -= len(scores)

        niter += 1

        if certain_table.lam != prev_lam or certain_table.mu != prev_mu:
            prev = time.time()
            select_heap.update_order()
            prev_lam = certain_table.lam
            prev_mu = certain_table.mu

        if niter % 10 == 0:
            print('\rIter-{}, Ri: min {}, max {}, mean {}, Pi: {:.3f}, Ei: {:.3f}'
                  .format(niter, certain_table.topk[0].s, certain_table.topk[-1].s, certain_table.topk_mean(), topk_prob, candidates[0].s if len(candidates)!=0 else 0), end="")
            sys.stdout.flush()


    print('\nIter-{}, Ri: min {}, max {}, mean {}, Pi: {:.3f}, Ei: {:.3f}'
          .format(niter, certain_table.topk[0].s, certain_table.topk[-1].s, certain_table.topk_mean(), topk_prob, candidates[0].s if len(candidates) != 0 else 0))

    topk = list(reversed(certain_table.topk))

    precision, rank_dist, score_error = evaluate(topk, k, lr, window, data_size)
    print("[EXP]precision:", precision) 
    print("[EXP]score_error:", score_error)
    print("[EXP]rank_dist:", rank_dist)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/virtualroad.data", help="path to data config file")
    parser.add_argument("--conf_thres", type=float, default=0.9)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--window_samples", type=int, default=1)
    parser.add_argument("--ref_dist", type=int, default=30)
    parser.add_argument("--score_func", choices=["count", "area", "area2", "depth"], default="count")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--use_cache", action='store_true')
    opt = parser.parse_args()
    print(opt)
    
    if opt.gpu is not None:
        config.device = torch.device("cuda:%d" % opt.gpu)
        config.decord_ctx = decord.gpu(opt.gpu)
    
    data_config = parse_data_config(opt.data_config)
    video_path = data_config['video']
    lmdb_path = data_config['lmdb']
    cached_gt_path = data_config['cached_gt']
    
    data_size = int(data_config["length"])
    split_path = data_config['split']
    offset = int(data_config["offset"])

    if opt.score_func != "depth":
        obj = int(data_config["object"])
        gt_thres = float(data_config["gt_thres"])
    
    if opt.score_func == "count":
        label_func = obj_count_label_func(obj, gt_thres)
        gtod_fps = 86
    elif opt.score_func == "area":
        label_func = vehicle_area_label_func(gt_thres)
        gtod_fps = 86
    elif opt.score_func == "area2":
        label_func = vehicle_area_label_func2(gt_thres)
        gtod_fps = 86
    elif opt.score_func == "depth":
        label_func = talgating_label_func
        gtod_fps = 112
    
    if opt.use_cache:
        label_reader = CachedGTLabelReader(cached_gt_path, offset)
    else:
        label_reader = LMDBLabelReader(lmdb_path, label_func, offset=offset)
    max_score = int(data_config["max_score"])
    topk(split_path,
            label_reader, 
            opt.k, 
            opt.conf_thres, 
            opt.window, 
            opt.window_samples, 
            opt.batch_size, 
            max_score, 
            opt.ref_dist,
            data_size,
            cached_gt_path,
            gtod_fps)
