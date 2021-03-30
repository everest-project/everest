import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import lmdb
import numpy as np
import decord
import os
import itertools
import operator
from decord import VideoReader
import config
import threading
import queue

LMDB_MAP_SIZE = 1 << 40
class DirectoryVideoReader():
    def __init__(self, img_dir, img_size=416):
        self.img_size = img_size
        img_paths = os.listdir(img_dir)
        img_paths = [im for im in img_paths if im.endswith(".jpg")]
        img_paths.sort(key=lambda x: (len(x), x))
        self.img_paths = [img_dir + "/" + im for im in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        pil_img = Image.open(self.img_paths[idx])
        img = transforms.ToTensor()(pil_img)
        pil_img.close()
        img = F.interpolate(img.unsqueeze(0), size=self.img_size, mode="nearest").squeeze(0)
        return img

    def get_batch(self, batch):
        return torch.stack([self.__getitem__(idx) for idx in batch])

class DecordVideoReader():
    def __init__(self, video_file, img_size=(416, 416), gpu=None, num_threads=8, offset=0, is_torch=True):
        self.is_torch = is_torch
        if is_torch:
            decord.bridge.set_bridge('torch')
        if type(img_size) is tuple:
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        self.offset = offset
        if gpu is None:
            ctx = config.decord_ctx
        else:
            ctx = decord.gpu(gpu)
        if type(img_size) == int:
            img_size = (img_size, img_size)
        self._vr = VideoReader(video_file, ctx=ctx, width=img_size[0], height=img_size[1], num_threads=num_threads)

    def __len__(self):
        return len(self._vr)-self.offset

    def __getitem__(self, idx):
        if self.is_torch:
            return self._vr[idx+self.offset].permute(2, 0, 1).contiguous().float().div(255)
        else:
            return self._vr[idx+self.offset].asnumpy()

    def get_batch(self, batch):
        batch = [b+self.offset for b in batch]
        if self.is_torch:
            return self._vr.get_batch(batch).permute(0, 3, 1, 2).contiguous().float().div(255)
        else:
            return self._vr.get_batch(batch).asnumpy()

def loader_thread(video_reader, label_reader, indices, batch_size, q):
    decord.bridge.set_bridge('torch') #need to set bridge in new thread
    length = (len(indices) + batch_size - 1) // batch_size

    for cur in range(length):
        start = cur * batch_size
        end = (cur + 1) * batch_size
        batch_idx = indices[start:end]
        imgs = video_reader.get_batch(batch_idx)
        if label_reader is not None:
            labels = label_reader.get_batch(batch_idx)
            labels = torch.tensor(labels)
            q.put((imgs, labels), block=True)
        else:
            q.put((imgs, None), block=True)

class VideoLoader(object):
    def __init__(self, video_reader, indices, label_reader=None, batch_size=64, cache=True, gpu=None):
        self._vr = video_reader
        self._lr = label_reader
        np.random.shuffle(indices)
        if gpu is None:
            self._device = config.device
        else:
            self._device = torch.device("cuda:%d" % gpu)
        self._indices = indices
        self._batch_size = batch_size
        self._queue = queue.Queue(128)
        self._cur = 0
        self._len = (len(indices) + batch_size - 1) // batch_size
        self._cache_imgs = torch.zeros([len(indices), 3, config.cmdn_input_size[0], config.cmdn_input_size[1]])
        self._cache_labels = torch.zeros([len(indices)])
        self._cached = False

    def __len__(self):
        return self._len

    def __iter__(self):
        self._cur = 0
        if not self._cached:
            threading.Thread(target=loader_thread, args=(self._vr, self._lr, self._indices, self._batch_size, self._queue)).start()
        return self

    def __next__(self):
        if self._cur >= self._len:
            self._cached = True
            raise StopIteration
        if not self._cached:
            imgs, labels = self._queue.get(block=True)
            self._cache_imgs[self._cur*self._batch_size:(1+self._cur)*self._batch_size] = imgs
            if labels is not None:
                self._cache_labels[self._cur*self._batch_size:(1+self._cur)*self._batch_size] = labels
        else:
            imgs = self._cache_imgs[self._cur*self._batch_size:(self._cur+1)*self._batch_size]
            if self._lr is not None:
                labels = self._cache_labels[self._cur*self._batch_size:(self._cur+1)*self._batch_size]
        self._cur += 1
        imgs = imgs.to(self._device)
        if labels is not None:
            labels = labels.to(self._device)
            return imgs, labels
        else:
            return imgs

def loader_diff_thread(video_reader, indices, diff_thres, batch_size, ref_dist, q):
    decord.bridge.set_bridge('torch') #need to set bridge in new thread
    length = (len(indices) + batch_size - 1) // batch_size
    assert batch_size % ref_dist == 0

    for cur in range(length):
        start = cur * batch_size
        end = (cur + 1) * batch_size
        batch_idx = indices[start:end]
        imgs = video_reader.get_batch(batch_idx).to(config.device)
        batch_idx = torch.tensor(batch_idx).to(config.device)
        
        discarded_ref_list = []
        remained_ref_list = []
        remained_list = []
        for i in range(batch_size // ref_dist):
            cut_idxs = batch_idx[i*ref_dist:(i+1)*ref_dist]
            cut_imgs = imgs[i*ref_dist:(i+1)*ref_dist]
            if len(cut_imgs) == 0:
                break
            ref_img = cut_imgs[len(cut_imgs) // 2]
            ref_idx = cut_idxs[len(cut_imgs) // 2]

            diff = ((cut_imgs - ref_img.unsqueeze(0))**2).view(len(cut_imgs), -1).mean(-1)

            discarded = ((diff < diff_thres) & (cut_idxs != ref_idx)).nonzero().squeeze(1)
            remained = ((diff >= diff_thres) | (cut_idxs == ref_idx)).nonzero().squeeze(1)

            remained_refs = torch.ones([len(remained), ref_dist + 1], dtype=torch.int32, device=config.device) * -1
            remained_refs[:,0] = cut_idxs[remained]
            remained_refs[:,1] = cut_idxs[remained]
            ref_offset = ((remained == len(cut_imgs) // 2).nonzero()).squeeze()
            remained_refs[ref_offset, 2:len(discarded)+2] = cut_idxs[discarded]

            refs = torch.ones([len(discarded)], dtype=torch.int64, device=config.device) * int(ref_idx)
            discarded_refs = torch.stack([cut_idxs[discarded], refs], 1)
            discarded_ref_list.append(discarded_refs)
            remained_ref_list.append(remained_refs)
            remained_list.append(remained)

        discarded_refs = torch.cat(discarded_ref_list, 0)
        remained_refs = torch.cat(remained_ref_list, 0)
        remained = torch.cat(remained_list, 0)
        imgs = imgs[remained]

        q.put((imgs, discarded_refs, remained_refs), block=True)

class VideoLoaderDiff(object):
    def __init__(self, video_reader, indices, diff_thres, batch_size=60, ref_dist=30):
        self._vr = video_reader
        self._indices = indices
        self._batch_size = batch_size
        self._len = (len(self._indices) + self._batch_size - 1) // self._batch_size
        self._diff_thres = diff_thres
        self._ref_dist = ref_dist
        self._queue = queue.Queue(128)

    def __len__(self):
        return self._len

    def __iter__(self):
        self._cur = 0
        threading.Thread(target=loader_diff_thread, args=(self._vr, self._indices, self._diff_thres, self._batch_size, self._ref_dist, self._queue)).start()
        return self

    def __next__(self):
        if self._cur >= self._len:
            raise StopIteration

        imgs, discarded_refs, remained_refs = self._queue.get()
        self._cur += 1
        return imgs, discarded_refs, remained_refs