import torch
import decord

LMDB_MAP_SIZE = 1 << 40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
decord_ctx = decord.gpu(0)

video_dir = "videos"
cached_gt_dir = "cached_gt"
cmdn_config = "config/cmdn.cfg"
cmdn_input_size = (128, 128)
