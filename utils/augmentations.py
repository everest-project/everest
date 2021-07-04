import torch
import torch.nn.functional as F
import numpy as np


def horizontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets
