"""Top-k operator."""
import copy
import heapq
import json
import logging
import numpy as np
import torch
import config as cfg
from collections import namedtuple
import os
import time
from scipy.special import logsumexp
from utils.parse_config import *
from config import *
from blist import sortedlist
from utils.topk_utils import *
import argparse
import random
import math
random.seed(0)

opt = None
path_list = []
idx_list = []
remained_ref = None
obj = None
certain_table = None
uncertain_table = None


SF = namedtuple('SF', ['s', 'f'])

def f2path(fs):
    return idx2path(path_list, f2idx(idx_list, fs))

class SelectionHeap():
    def __init__(self, batch_size):
        self.size = len(uncertain_table.log_cdf)
        self.batch_size = batch_size
        if certain_table.lam == 0:
            self.LF = np.zeros([self.size])
        else:
            self.LF = self.compute_all_L()
        self.order = np.argsort(self.LF)[::-1]

    def compute_all_L(self):
        log_cdf = uncertain_table.log_cdf
        return np.log(1 - np.exp(log_cdf[:,certain_table.lam])) - log_cdf[:,certain_table.mu]

    def compute_L(self, f):
        log_cdf = uncertain_table.log_cdf[f]
        return np.log(1 - np.exp(log_cdf[certain_table.lam])) - log_cdf[certain_table.mu]

    def compute_E(self, f, topk_prob):
        lam = certain_table.lam
        mu = certain_table.mu
        log_cdf = uncertain_table.log_cdf[f]
        log_pdf = uncertain_table.log_pdf[f]
        terms = np.zeros([2 + mu-lam])
        terms[0] = topk_prob
        terms[-1] = uncertain_table.cum(mu) - log_cdf[mu] + np.log(1 - np.exp(log_cdf[mu]))
        mid = np.arange(lam+1, mu+1)
        terms[1:-1] = uncertain_table.cum(mid) - log_cdf[mid] + log_pdf[mid]
        return logsumexp(terms)

    def update_order(self):
        self.LF = self.compute_all_L()
        self.order = np.argsort(self.LF)[::-1]

    def bootstrap(self, size):
        return np.argsort(uncertain_table.log_cdf[:,0])[:size]

    def select(self, topk_prob):
        candidate_heap = PQueue(self.size)
        gamma = uncertain_table.cum(certain_table.mu)
        idx = 0
        while idx < self.size:
            f = self.order[idx]
            if certain_table.is_certain(f):
                idx += 1
                continue
            E = self.compute_E(f, topk_prob)
            candidate_heap.push(SF(E, f))
            U = logsumexp([topk_prob, self.LF[self.order[idx]] + gamma])
            if candidate_heap.size >= self.batch_size and candidate_heap.max(1).s > U:
                break;
            idx += 1
        return candidate_heap.top(min(self.size, self.batch_size))

class UncertainTable():
    def __init__(self, cdf):
        self.log_cdf = np.log(cdf)
        self.H = self.log_cdf.sum(0)
        self.log_pdf = np.zeros(cdf.shape)
        self.log_pdf[:,0] = cdf[:,0]
        self.log_pdf[:,1:] = cdf[:,1:] - cdf[:,:-1]
        self.log_pdf = np.log(self.log_pdf)

    def cum(self, s):
        return self.H[s] - certain_table.H[s]

class CertainTable():
    def __init__(self, k, initial=None):
        initial_sfs = []
        if initial is not None:
            scores = groundtruth_shortcut(idx2path(path_list, initial), obj, opt.score_func)
            initial_sfs = [SF(s, f) for (s,f) in zip(scores, initial)]
            initial_sfs.sort(reverse=True)
        self.topk = sortedlist(initial_sfs[:k])
        self.lam = 0 if len(self.topk) == 0 else self.topk[0].s
        self.mu = self.lam if len(self.topk) <= 1 else self.topk[1].s
        self.H = np.zeros([cdf.shape[1]])
        self.bitmap = np.zeros([cdf.shape[0]], dtype='bool')
        self.k = k

    def insert_sw(self, sw):
        self.insert_mirror(SF(sw.s, sw.f))
        self.H += uncertain_table.log_cdf[sw.f]
        self.bitmap[sw.f] = True

    def insert_sf(self, sf):
        self.insert_mirror(SF(sf.s, f2idx(idx_list, [sf.f])[0]))
        self.H += uncertain_table.log_cdf[sf.f]
        self.bitmap[sf.f] = True

    def insert_mirror(self, s_idx):
        self.topk.add(s_idx)
        if len(self.topk) > self.k:
            self.topk.pop(0)
        self.lam = 0 if len(self.topk) == 0 else self.topk[0].s
        self.mu = self.lam if len(self.topk) <= 1 else self.topk[1].s

    def is_certain(self, f):
        return self.bitmap[f]

    def topk_prob(self):
        return uncertain_table.cum(self.lam)

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

def precision(approx, exact):
    k = len(exact)
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
    k = len(exact)
    rank = np.zeros((k,))
    # assume rank k+1 if missing element
    rank[:] = k
    j = 0
    for i in range(k):
        if j > k-1:
            break
        while approx[i] != exact[j] and j < k-1:
            j += 1
        if approx[i] == exact[j]:
            rank[i] = j
            j += 1
    dis = np.sum(np.abs(np.arange(k) - rank))
    return dis / k

def score_error(approx, exact):
    k = len(exact)

    approx_ = np.array(approx)
    exact_ = np.array(exact)

    err = np.sum(np.abs(approx_ - exact_))

    return err / k

def evaluate(topk, k, window=1):
    topk_scores = [sf.s for sf in topk]
    topk_idxs = [sf.f for sf in topk]
    print("finding ground-truth label")
    if window < 2:
        gts = np.array(groundtruth_shortcut(path_list, obj, opt.score_func))
    else:
        num_windows = math.ceil(len(path_list) / window)
        gts = np.array(groundtruth_window(range(0, num_windows), window, window))
    print("sorting ground-truth")
    gt_idxs = gts.argsort()[::-1][:k]
    gt_scores = gts[gt_idxs]

    prec = precision(topk_scores, gt_scores)
    rd = rank_distance(topk_scores, gt_scores)
    se = score_error(topk_scores, gt_scores)

    print('----------------------------------------')
    print('Top-{} value: {}'.format(k, topk_scores))
    print('Top-{} gt value: {}'.format(k, gt_scores))
    print('Top-{} indices: {}'.format(k, topk_idxs))
    print('Top-{} gt indices: {}'.format(k, gt_idxs))
    print('Precision: {}'.format(prec))
    print('Rank Distance: {}'.format(rd))
    print('Score Error: {}'.format(se))

def groundtruth_window(ws, window_size, window_samples):
    scores = []
    for w in ws:
        paths = path_list[w * window_size: (w+1) * window_size]
        random.shuffle(paths)
        window_scores = groundtruth_shortcut(paths[:window_samples], obj, opt.score_func)
        scores.append(int(round(np.mean(window_scores))))
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/virtualroad.data", help="path to data config file")
    parser.add_argument("--conf_thres", type=float, default=0.9)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--window_samples", type=int, default=1)
    opt = parser.parse_args()
    print(opt)

    prev = time.time()
    data_config = parse_data_config(opt.data_config)
    split_dir = data_config["split"]
    obj = int(data_config["object"])
    distribution_dir = data_config["distribution"]

    with open(split_dir + "/all.txt", "rt") as f:
        path_list = f.readlines()
        path_list = [p.rstrip() for p in path_list]

    cdf = np.load(distribution_dir + "/cdf.npy")
    uncertain_table = UncertainTable(cdf)

    if opt.window < 2:
        remained_ref = np.load(split_dir + "/remained.npy")
        train_idx = np.load(split_dir + "/train.npy")
        val_idx = np.load(split_dir + "/val.npy")
        initial_certain = np.concatenate([train_idx, val_idx], 0)
        idx_list = remained_ref[:,0]
        certain_table = CertainTable(opt.k, initial_certain)
    else:
        certain_table = CertainTable(opt.k)

    select_heap = SelectionHeap(opt.batch_size)
    niter = 0
    prev_lam, prev_mu = -1, -1
    while select_heap.size > 0:
        topk_prob = certain_table.topk_prob()
        if topk_prob >= np.log(opt.conf_thres):
            break

        if len(certain_table.topk) < opt.k:
            print("bootstraping")
            clean_f = select_heap.bootstrap(opt.k - len(certain_table.topk))
        else:
            candidates = select_heap.select(topk_prob)
            clean_f = [sf.f for sf in candidates]
        if opt.window < 2:
            scores = groundtruth_shortcut(f2path(clean_f), obj, opt.score_func)
            for f, score in zip(clean_f, scores):
                certain_table.insert_sf(SF(score, f))
                mirrors = remained_ref[f, 2:]
                for m in mirrors:
                    if m == -1:
                        break
                    certain_table.insert_mirror(SF(score, m))
        else:
            scores = groundtruth_window(clean_f, opt.window, opt.window_samples)
            for w, score in zip(clean_f, scores):
                certain_table.insert_sw(SF(score, w))

        niter += 1

        if certain_table.lam != prev_lam or certain_table.mu != prev_mu:
            select_heap.update_order()
            prev_lam = certain_table.lam
            prev_mu = certain_table.mu

        if niter % 10 == 0:
            print('Iter-{}, Ri: min {}, max {}, mean {}, Pi: {:.3f}, Ei: {:.3f}'
                  .format(niter, certain_table.topk[0].s, certain_table.topk[-1].s, certain_table.topk_mean(), topk_prob, candidates[0].s))

    print('Iter-{}, Ri: min {}, max {}, mean {}, Pi: {:.3f}, Ei: {:.3f}'
          .format(niter, certain_table.topk[0].s, certain_table.topk[-1].s, certain_table.topk_mean(), topk_prob, candidates[0].s))

    topk = list(reversed(certain_table.topk))

    cur = time.time()
    print("algorithm runtime:", cur - prev)
    print("inference runtime:", niter * opt.batch_size / 30)
    print("total runtime:", cur - prev + niter * opt.batch_size / 30)
    evaluate(topk, opt.k, opt.window)

