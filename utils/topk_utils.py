import numpy as np
import os
def f2idx(idx_list, fs):
    return [idx_list[f] for f in fs]

def idx2path(path_list, idxs):
    return [path_list[i] for i in idxs]

def groundtruth_shortcut(img_paths, obj, score_func):
    """
    Only for debug purpose
    """
    scores = []
    for p in img_paths:
        p = p.replace("images_resize", "labels").replace(".jpg", ".txt")
        targets = []
        if os.stat(p).st_size != 0:
            targets = np.loadtxt(p, delimiter=',').reshape(-1, 7)
        if score_func == "count":
            targets = [t for t in targets if int(t[-1]) == obj]
            score = len(targets)
        elif score_func == "area":
            bus_idx = 5
            targets = [t for t in targets if int(t[-1]) == obj or int(t[-1]) == bus_idx]
            score = 0
            for det in targets:
                score += (float(det[2]) - float(det[0])) * (float(det[3]) - float(det[1]))
            score /= 416 * 416
            score = int(min(score, 1) * 50)
        else:
            raise NotImplementedError
        scores.append(score)
    return scores

