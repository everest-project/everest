import numpy as np
import os
def f2idx(idx_list, fs):
    return [idx_list[f] for f in fs]

def idx2path(path_list, idxs):
    return [path_list[i] for i in idxs]

def groundtruth_shortcut(img_paths, obj):
    """
    Only for debug purpose
    """
    scores = []
    for p in img_paths:
        p = p.replace("images_resize", "labels").replace(".jpg", ".txt")
        targets = []
        if os.stat(p).st_size != 0:
            targets = np.loadtxt(p, delimiter=',').reshape(-1, 7)
        targets = [t for t in targets if int(t[-1]) == obj]
        scores.append(len(targets))
    return scores

