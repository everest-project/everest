import numpy as np
import os
import lmdb
import config
from models import *
from utils.utils import *
from utils.parse_config import parse_model_config
import math
import io
from PIL import Image
import numpy as np
import torch
import cv2

LMDB_MAP_SIZE = 1 << 40

class LMDBLabelReader():
    def __init__(self, lmdb_dir, label_func, kernel_size=1, filter_func=None, offset=0):
        self.label_func = label_func
        self.filter_func = filter_func
        self.kernel_size = kernel_size
        self.offset = offset
        self.env = lmdb.open(lmdb_dir, map_size=LMDB_MAP_SIZE, lock=False)
        with self.env.begin(write=False) as txn:            
            len_bytes = txn.get('label_len'.encode())
            if (len_bytes!=None):
                self._len = int.from_bytes(len_bytes, 'big')
            else:
                self._len = txn.stat()['entries']

    def __len__(self):
        return self._len - self.offset

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            return self._get_item(idx + self.offset, txn)

    def _get_item(self, idx, txn):
        range_min = max(0, idx - self.kernel_size // 2)
        range_max = min(self._len, idx + self.kernel_size // 2 + 1)
        batch = range(range_min, range_max)
        labels = []
        for idx in batch:
            boxes_bytes = txn.get(("label_%d" % idx).encode())
            if self.label_func == talgating_label_func:
                labels.append(self.label_func(boxes_bytes))
            else:
                boxes = np.frombuffer(boxes_bytes, dtype=np.float32).reshape(-1, 7).copy()
                labels.append(self.label_func(boxes))
        if self.filter_func is not None:
            return self.filter_func(np.array(labels))
        else:
            return labels[0]

    def get_batch(self, batch):
        labels = []
        with self.env.begin(write=False) as txn:
            for idx in batch:
                labels.append(self._get_item(idx + self.offset, txn))
        return labels

class CachedGTLabelReader():
    def __init__(self, cached_gt_path, offset=0):
        self.cached_gt = np.load(cached_gt_path)
        self.offset = offset

    def __len__(self):
        return len(self.cached_gt)

    def __getitem__(self, idx):
        return self.cached_gt[idx + self.offset]

    def get_batch(self, batch):
        batch = [idx + self.offset for idx in batch]
        return self.cached_gt[batch]

class LabelReader():
    def __init__(self, vr, udf, offset=0):
        self.vr = vr
        self.udf = udf
        self.offset = 0
        self.labels = {}
    
    def __len__(self):
        #return len(self.labels)
        return len(self.vr)

    def __getitem__(self, idx):
        if (self.labels.get(idx+self.offset)==None):            
            img = self.vr[idx+self.offset]
            
            # transform tensor to numpy
            if type(img) is torch.Tensor:
                img = (img.permute(1,2,0)*255).cpu().numpy().astype('uint8')
            
            # image height and width should be (416,416)
            img = img[np.newaxis,:]
            self.labels[idx + self.offset] = self.udf.get_scores(img)[0]
        return self.labels.get(idx+self.offset)

    def get_batch(self, batch):
        labels = []
        for idx in batch:
            labels.append(self[idx])
        return labels

 
