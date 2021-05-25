import sys
import os
import io

import matplotlib

from oracle.udf.base import BaseScoringUDF
import config
sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import Image

from utils.parse_config import parse_model_config
from models import *
from utils.utils import *
import cv2
import numpy as np

matplotlib.use('agg')


#obj_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
obj_names = ["ball", "door"]

class ExcitingMoment(BaseScoringUDF):
    def __init__(self):
        super().__init__()
        self.arg_parser.add_argument("--class_thres", type=float, default=0.5)
        self.arg_parser.add_argument("--obj_thres", type=float, default=0)
        self.arg_parser.add_argument("--obj", type=str, choices=obj_names, default="ball")
        # self.model_config = 'config/yolov3.cfg'
        # self.weights = 'weights/yolov3.weights'        
        self.model_config = 'config/yolov3-spp5-custom.cfg'
        self.weights = 'weights/yolov3-spp5-custom_best.weights'
    
    def initialize(self, opt, gpu=None):
        self.opt = opt
        self.obj = obj_names.index(opt.obj)
        self.device = config.device
        if gpu is not None:
            self.device = torch.device(f"cuda:{gpu}")
        self.model = Darknet(parse_model_config(self.model_config)).to(self.device)
        self.model.load_darknet_weights(self.weights)
        self.model.eval()
    
    def get_img_size(self):
        return (416, 416)
    
    def get_scores(self, imgs, visualize=False):
        assert (imgs.shape[1], imgs.shape[2]) == self.get_img_size()
        model_imgs = torch.from_numpy(imgs).float().to(self.device)
        model_imgs = model_imgs.permute(0, 3, 1, 2).contiguous().div(255)

        with torch.no_grad():
            detections = self.model(model_imgs)
            detections = non_max_suppression(detections, 0.1, 0.45)
        scores = []
        visual_imgs = []
        
        #start change 
            if boxes is None:
                scores.append(0)
                if visualize:
                    visual_imgs.append(imgs[i])
            else:
                for i, boxes in enumerate(detections):
                    boxes = [box for box in boxes if float(box[4]) >= self.opt.class_thres and float(box[5]) >= self.opt.obj_thres]
                    boxes = [box for box in boxes if int(box[-1]) == 0 or int(box[-1]) == 1]

                ball_exist = False
                door_exist = False

                ball_max_conf = 0.0
                door_max_conf = 0.0

                ball_max = None
                door_max = None

                score = 0
                
                for box in boxes:
                    if box[-1] == 0:
                        ball_exist = True
                        if box[4] > ball_max_conf:
                            ball_max_conf = box[4]
                            ball_max = box
                    if box[-1] == 1:
                        door_exist = True
                        if box[4] > door_max_conf:
                            door_max_conf = box[4]
                            door_max = box
                if ball_exist == True and door_exist == True:
                    ball_x = (ball_max[0]+ball_max[2])/2
                    ball_y = (ball_max[1]+ball_max[3])/2
                    door_x = (door_max[0]+door_max[2])/2
                    door_y = (door_max[1]+door_max[3])/2
                    score = 1000-pow(math.sqrt(pow((door_x-ball_x),2)+pow((door_y-ball_y),2)),2)
                    score = int(score/20)
                    print(score)
                    if score <= 0:
                        score = 0
                scores.append(score)
                if visualize:
                    visual_img = np.copy(imgs[i])
                    visual_img = cv2.resize(visual_img, (739, 416))
                    # draw ball rectangle          
                    x1 = int(ball_max[0].item() / 416 * 739)
                    x2 = int(ball_max[2].item() / 416 * 739)
                    y1 = int(ball_max[1].item())
                    y2 = int(ball_max[3].item())
                    cv2.rectangle(visual_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # draw door rectangle
                    x1 = int(door_max[0].item() / 416 * 739)
                    x2 = int(door_max[2].item() / 416 * 739)
                    y1 = int(door_max[1].item())
                    y2 = int(door_max[3].item())
                    cv2.rectangle(visual_img, (x1, y1), (x2, y2), (0, 255, 0), 2)                    
                if visual_imgs:
                    visual_imgs.append(Image.fromarray(visual_img))
                
        # end change        
        if visualize:
            return scores, visual_imgs
        else:
            return scores
