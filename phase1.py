import os
import sys
import time
import random
import argparse
import numpy as np
import math
import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.distributions import normal
import torch.optim as optim

import config
from utils.topk_utils import *
from utils.label_reader import *
from utils.parse_config import *
from utils.video_reader import *


def split_dataset(opt, vr, lr, save=False):
    random.seed(opt.random_seed)
    length = get_video_length(opt, vr)
    indices = list(range(length))
    num_train, num_valid = get_train_valid_size(opt, vr)
    random.shuffle(indices)
    train = indices[:num_train]
    valid = indices[num_train:num_train+num_valid]
    test = indices[num_train+num_valid:]

    train.sort()
    test.sort()
    valid.sort()

    # compute the label weight for training CMDN
    scores = lr.get_batch(train + valid)
    hist = np.ones([opt.max_score]) * 10 # to smoothn the weight
    for s in scores:
        hist[s] += 1

    sample_max = hist.max()
    weight = sample_max / hist

    train = np.array(train)
    valid = np.array(valid)
    test = np.array(test)
    weight = np.array(weight)

    if save:
        split_path = get_split_path(opt)
        os.makedirs(split_path, exist_ok=True)
        np.save(os.path.join(split_path, "train_idxs.npy"), train)
        np.save(os.path.join(split_path, "valid_idxs.npy"), valid)
        np.save(os.path.join(split_path, "test_idxs.npy"), test)
        np.save(os.path.join(split_path, "score_weight.npy"), weight)

    return train, valid, test, weight

def train_cmdn(opt, vr, lr, train_idxs, valid_idxs, score_weight):
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(opt.random_seed)

    # Get data configuration
    model_group_config = parse_model_config(config.cmdn_config)
    model_configs = parse_model_group(model_group_config)
    checkpoint_dir = get_checkpoint_dir(opt)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_video_loader = VideoLoader(vr, train_idxs, lr, batch_size=opt.cmdn_train_batch)
    valid_video_loader = VideoLoader(vr, valid_idxs, lr, batch_size=opt.cmdn_train_batch)

    nlls = np.zeros([len(model_configs)])
    model_batch = len(model_configs)
    for i in range(int(math.ceil(len(model_configs) / model_batch))):
        model_config_batch = model_configs[i*model_batch: (i+1) * model_batch]
        nlls[i*model_batch: (i+1)*model_batch] = train_models(opt.cmdn_train_epochs, model_config_batch, range(i*model_batch, (i+1)*model_batch), train_video_loader, valid_video_loader, score_weight, checkpoint_dir)
    best_model = np.argmin(nlls)

    print("best_model: %d, nll: %0.3f" % (best_model, nlls[best_model]))
    best_model_path = os.path.join(checkpoint_dir, f"cmdn_{best_model}_best.pth")
    os.rename(os.path.join(checkpoint_dir, f"cmdn_{best_model}.pth"), best_model_path)
    return best_model_path

def train_models(epochs, model_configs, mids, train_dataloader, valid_dataloader, weight, checkpoint_dir):
    for i, model_config in enumerate(model_configs):
        for module_def in model_config:
            if module_def["type"] == "hmdn":
                print("Model_%d: M: %s, H: %s, eps: %s" % (mids[i], module_def["M"], module_def["num_h"], module_def["eps"]))
                break

    models = [Darknet(model_config, weight).to(config.device) for model_config in model_configs]
    parameters = []
    for model in models:
        model.apply(weights_init_normal)
        model.load_darknet_weights("weights/yolov3-tiny.weights")
        model.train()
        parameters += model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        num_batchs = 0
        for imgs, scores in tqdm.tqdm(train_dataloader, desc="epoch %d/%d" % (epoch+1, epochs)):
            loss = 0.0
            for model in models:
                _, mdn_output = model(imgs, scores=scores)
                wta_loss = mdn_output[0]
                mdn_loss = mdn_output[1]
                if epoch < 5:
                    loss += wta_loss + 0.0 * mdn_loss
                else:
                    loss += 0.5 * wta_loss + mdn_loss

                # mdn metric
                total_loss += mdn_loss.item()
                num_batchs += 1
                model.seen += imgs.size(0)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("mdn_loss: %.3f" % (total_loss / num_batchs))

    nlls = evaluate_video(models, mids, valid_dataloader)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"cmdn_{mids[i]}.pth"))
    return np.array(nlls)

def evaluate_video(models, mids, dataloader):
    for model in models:
        model.eval()
    mdn_loss_list = [0] * len(models)
    num_loss_list = [0] * len(models)
    mean_dict_list = [{} for m in models]
    var_dict_list = [{} for m in models]
    for imgs, scores in tqdm.tqdm(dataloader, desc="Detecting objects"):
        scores_cpu = scores.cpu()
        with torch.no_grad():
            for i, (mean_dict, var_dict, model) in enumerate(zip(mean_dict_list, var_dict_list, models)):
                _, mdn_output = model(imgs, scores=scores)
                pi, sigma, mu = mdn_output[2], mdn_output[3], mdn_output[4]
                mdn_loss_list[i] += mdn_output[1].item()
                num_loss_list[i] += 1
                mean = (mu * pi).sum(-1)
                var = ((sigma**2 + mu**2 - mean.unsqueeze(-1)**2) * pi).sum(-1)
                for i in range(len(mean)):
                    lab = scores_cpu[i].item()
                    if lab not in mean_dict:
                        mean_dict[lab] = [mean[i].item()]
                        var_dict[lab] = [var[i].item()]
                    else:
                        mean_dict[lab] += [mean[i].item()]
                        var_dict[lab] += [var[i].item()]

    for i, (mean_dict, var_dict) in enumerate(zip(mean_dict_list, var_dict_list)):
        print("profile of Model %d" % mids[i])
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
        print('NLL: {:.2f}'.format(mdn_loss_list[i] / num_loss_list[i]))
        mdn_loss_list[i] /= num_loss_list[i]
    return mdn_loss_list

def cmdn_scan(opt, best_model_path, vr, test_idxs, save=False):
    prefix = os.path.splitext(os.path.basename(best_model_path))[0]
    model_id = int(prefix.split("_")[1])
    print(model_id)
    model_group_config = parse_model_config(config.cmdn_config)
    model_configs = parse_model_group(model_group_config)   
    cmdn_config = model_configs[model_id]

    cmdn = Darknet(cmdn_config).to(config.device)
    cmdn.load_state_dict(torch.load(best_model_path, map_location=config.device))

    dataloader = VideoLoaderDiff(vr, test_idxs, opt.diff_thres, opt.cmdn_scan_batch, opt.cmdn_scan_batch // 2)
    cmdn.eval()

    total_pi_list = []
    total_sigma_list = []
    total_mu_list = []
    discarded_list = []
    remained_list = []

    with torch.no_grad():
        for imgs, discarded, remained in tqdm.tqdm(dataloader, desc="Inferencing"):
            _, mdn_output = cmdn(imgs) 
            pi, sigma, mu = mdn_output[2], mdn_output[3], mdn_output[4]
            total_pi_list.append(pi.cpu())
            total_sigma_list.append(sigma.cpu())
            total_mu_list.append(mu.cpu())
            discarded_list.append(discarded.cpu())
            remained_list.append(remained.cpu())

    discarded = torch.cat(discarded_list, 0).numpy().astype(np.int32)
    remained = torch.cat(remained_list, 0).numpy().astype(np.int32)
    total_pi = torch.cat(total_pi_list, 0).numpy().astype(np.float32)
    total_sigma = torch.cat(total_sigma_list, 0).numpy().astype(np.float32)
    total_mu = torch.cat(total_mu_list, 0).numpy().astype(np.float32)

    if save:
        split_path = get_split_path(opt)
        np.save(os.path.join(split_path, "mu.npy"), total_mu)
        np.save(os.path.join(split_path, "sigma.npy"), total_sigma)
        np.save(os.path.join(split_path, "pi.npy"), total_pi)
        np.save(os.path.join(split_path, "discarded.npy"), discarded)
        np.save(os.path.join(split_path, "remained.npy"), remained)

    return total_mu, total_sigma, total_pi, discarded, remained

def window_distribution(opt, train_idxs, valid_idxs, mu, sigma, pi, discarded_ref, remained_ref, lr, vr):
    data_size = get_video_length(opt, vr)
    ref_dist = opt.cmdn_scan_batch // 2
    window = opt.window
    num = len(remained_ref)
    
    remained_idx = remained_ref[:,0]
    num_mixtures = pi.shape[1]

    train_scores = lr.get_batch(train_idxs)
    val_scores = lr.get_batch(valid_idxs)
    
    num = data_size
    pi_all = np.zeros([num, num_mixtures], dtype=np.float32)
    sigma_all = np.zeros([num, num_mixtures], dtype=np.float32)
    mu_all = np.zeros([num, num_mixtures], dtype=np.float32)

    pi_all[train_idxs, 0] = 1.0
    mu_all[train_idxs, 0] = train_scores
    sigma_all[train_idxs, :] = 0.1
    pi_all[valid_idxs, 0] = 1.0
    mu_all[valid_idxs, 0] = val_scores
    sigma_all[valid_idxs, :] = 0.1
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

def gen_cdf(pi, mu, sigma, max_score, batch_size=5000):
    num = len(pi)
    batch_size = int(min(num, batch_size))

    cdf_list = []
    ticks = torch.arange(0.5, max_score + 0.5, 1, device=config.device).view(1, 1, max_score)
    for b in range(math.ceil(num / batch_size)):
        pi_gpu = torch.from_numpy(pi[b * batch_size: (b+1) * batch_size]).to(config.device)
        mu_gpu = torch.from_numpy(mu[b * batch_size: (b+1) * batch_size]).to(config.device)
        sigma_gpu = torch.from_numpy(sigma[b * batch_size: (b+1) * batch_size]).to(config.device)
        normals = normal.Normal(mu_gpu.unsqueeze(-1), sigma_gpu.unsqueeze(-1))
        cdf = normals.cdf(ticks)
        cdf = (cdf * pi_gpu.unsqueeze(-1)).sum(1)
        cdf = torch.where(cdf >= 0.997, torch.ones(cdf.shape, device=config.device), cdf)
        cdf = cdf.clamp(0, 1)
        cdf_list.append(cdf.cpu().numpy())
        torch.cuda.empty_cache()
    cdf = np.concatenate(cdf_list, 0)
    return cdf.astype(np.float64)