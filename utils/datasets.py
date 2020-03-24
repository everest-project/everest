import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)

class VideoObjectDataset_Diff(Dataset):
    def __init__(self, test_path, test_idx,img_size=416, threshold=0.01,size=None, obj=2, ref_dist=20):
        with open(test_path, "r") as file:
            self.img_files = file.readlines()

        if size != None:
            self.img_files = self.img_files[:size]

        self.img_files = [img_file.rstrip() for img_file in self.img_files]
        self.test_idx = np.load(test_idx)

        self.threshold = threshold
        self.img_size = img_size
        self.batch_count = 0
        self.ref_dist = ref_dist

    def __getitem__(self, index):
        img_path = self.img_files[index]

        # Extract image as PyTorch tensor
        pl_img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(pl_img)
        pl_img.close()
        img = resize(img, self.img_size)
        idx = self.test_idx[index]

        return torch.tensor(idx), img

    def collate_fn(self, batch):
        idxs, imgs = list(zip(*batch))
        ref_img = imgs[len(imgs) // 2]
        ref_idx = idxs[len(imgs) // 2]

        imgs = torch.stack(imgs)
        idxs = torch.stack(idxs)
        diff = ((imgs - ref_img.unsqueeze(0))**2).view(len(imgs), -1).mean(-1)

        discarded = ((diff < self.threshold) & (idxs != ref_idx)).nonzero().squeeze(1)
        remained = ((diff >= self.threshold) | (idxs == ref_idx)).nonzero().squeeze(1)

        remained_refs = torch.ones([len(remained), self.ref_dist + 1], dtype=torch.int64) * -1
        remained_refs[:,0] = idxs[remained]
        remained_refs[:,1] = idxs[remained]
        ref_offset = ((remained == len(imgs) // 2).nonzero()).squeeze()
        remained_refs[ref_offset, 2:len(discarded)+2] = idxs[discarded]

        imgs = imgs[remained]
        refs = torch.ones([len(discarded)], dtype=torch.int64) * int(ref_idx)
        discarded_refs = torch.stack([idxs[discarded], refs], 1)
        self.batch_count += 1
        return imgs, discarded_refs, remained_refs


    def __len__(self):
        return len(self.img_files)


class VideoObjectDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, size=None, obj=2, score_func="count"):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        if size != None:
            #random.shuffle(self.img_files)
            self.img_files = self.img_files[:size]

        self.img_files = [img_file.rstrip() for img_file in self.img_files]

        self.label_files = [
            path.replace("images_resize", "labels").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.obj = obj
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.batch_count = 0
        self.score_func = score_func

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)]

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        img = resize(img, self.img_size)

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        boxes = []
        if os.stat(label_path).st_size != 0:
            boxes = np.loadtxt(label_path, delimiter=',').reshape(-1, 7)
            if self.score_func == "count":
                boxes = [box for box in boxes if int(box[-1]) == self.obj]
            elif self.score_func == "area":
                bus_idx = 5
                boxes = [box for box in boxes if int(box[-1]) == self.obj or int(box[-1]) == bus_idx]
            else:
                raise NotImplementedError

        if len(boxes) == 0:
            boxes = np.array([])
        else:
            boxes = np.stack(boxes)
        ratio = self.img_size / 416
        targets = None
        if boxes.shape[0] > 0:
            boxes = torch.from_numpy(boxes)
            targets = torch.zeros((len(boxes), 6))
            targets[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / self.img_size * ratio
            targets[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / self.img_size * ratio
            targets[:, 4] = (boxes[:, 2] - boxes[:, 0]) / self.img_size * ratio
            targets[:, 5] = (boxes[:, 3] - boxes[:, 1]) / self.img_size * ratio

            if self.augment:
                if np.random.random() < 0.5:
                    img, targets = horisontal_flip(img, targets)

        if self.score_func == "count":
            score = len(boxes)
        elif self.score_func == "area":
            score = 0
            for det in boxes:
                score += (float(det[2]) - float(det[0])) * (float(det[3]) - float(det[1]))
            score /= 416 * 416
            score = int(min(score, 1) * 50)
        else:
            raise NotImplementedError

        return img_path, img, targets, score

    def collate_fn(self, batch):
        paths, imgs, targets, scores = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        if len(targets) == 0:
            targets = torch.FloatTensor([]).view(0, 7)
        else:
            targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        imgs = torch.stack(imgs)
        scores = torch.FloatTensor(scores)
        self.batch_count += 1
        return paths, imgs, targets, scores

    def __len__(self):
        return len(self.img_files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
