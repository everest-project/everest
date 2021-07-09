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

from utils.happy_moment_tools import sentiment_analysis 
from utils.alexnet import KitModel as AlexNet
from utils.vgg19 import KitModel as VGG19


matplotlib.use('agg')


obj_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

class HappyMoment(BaseScoringUDF):
    def __init__(self):
        super().__init__()
        self.arg_parser.add_argument("--class_thres", type=float, default=0.5)
        self.arg_parser.add_argument("--obj_thres", type=float, default=0)
        self.arg_parser.add_argument("--obj", type=str, choices=obj_names, default="person")
        self.script_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.model_config =os.path.join(self.script_path, 'config/yolov3.cfg')
        self.weights =os.path.join(self.script_path, 'weights/yolov3.weights')
    
    def initialize(self, opt, gpu=None,sentiment_model_name = 'vgg19_finetuned_all'):
        self.opt = opt
        self.obj = obj_names.index(opt.obj)
        self.device = config.device
        if gpu is not None:
            self.device = torch.device(f"cuda:{gpu}")
        self.model = Darknet(parse_model_config(self.model_config)).to(self.device)
        self.model.load_darknet_weights(self.weights)
        self.model.eval()
        
        print("Loading sentiment analysis weights...")
        self.sentiment_model = AlexNet if 'hybrid' in sentiment_model_name else VGG19
        self.sentiment_model = self.sentiment_model(os.path.join(self.script_path,'weights/{}.pth'.format(sentiment_model_name))).to('cuda')
        self.sentiment_model.eval()
        print("Done...")
    
    def get_img_size(self):
        return (416, 416)
    
    def get_scores(self, imgs, visualize=False):
        assert (imgs.shape[1], imgs.shape[2]) == self.get_img_size()
        model_imgs = torch.from_numpy(imgs).float().to(self.device)
        model_imgs = model_imgs.permute(0, 3, 1, 2).contiguous().div(255)

        with torch.no_grad():
            detections = self.model(model_imgs)
            detections = non_max_suppression(detections, 0.1, 0.45)


        scores = [0 for i in range(len(imgs))]
        visual_imgs = []
        record = {}
        crop_idx = 0
        frame_idx = 0
        cropped_imgs = []
        
        for i, boxes in enumerate(detections):
            if boxes is None:
                if visualize:
                    visual_imgs.append(imgs[i])
            else:
                relavant_boxes = [box for box in boxes if int(box[-1]) == self.obj and float(box[4]) >= self.opt.class_thres and float(box[5]) >= self.opt.obj_thres]
                
                visual_img = np.copy(imgs[i])
                visual_img = cv2.resize(visual_img, (739, 416))
                
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in relavant_boxes:
                    x1 = int(x1.item() / 416 * 739)
                    if x1<0:
                        x1=0

                    x2 = int(x2.item() / 416 * 739)
                    y1 = int(y1.item())
                    
                    if y1<0:
                        y1 =0
                    y2 = int(y2.item())
                                           
                    cropped_imgs.append(visual_img[y1:(y2+1), x1:(x2+1)])                
                    record[crop_idx]= frame_idx
                    crop_idx += 1
                    
                    cv2.rectangle(visual_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                if visualize:
                    visual_imgs.append(Image.fromarray(visual_img))
            frame_idx += 1
        if cropped_imgs:
            crop_scores = sentiment_analysis(cropped_imgs, self.sentiment_model)
        
            for i, crop_score in enumerate(crop_scores):
                scores[record[i]] += crop_score
            
        if visualize:
            return scores, visual_imgs
        else:
            return scores
            
            
            
            
