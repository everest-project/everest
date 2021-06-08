# refer to furkanc/Yolov3-Face-Recognition
# I've used InsightFace_Pytorch for face recognition.
import sys
import os
import io

import cv2
import argparse
import numpy as np

from oracle.udf.base import BaseScoringUDF
import config
sys.path.insert(0, os.path.dirname(__file__))


import torch
from PIL import Image
from utils.utils import *

from utils.happy_moment_tools import sentiment_analysis 
from alexnet import KitModel as AlexNet
from vgg19 import KitModel as VGG19
from tqdm import tqdm




class HappyMoment(BaseScoringUDF):
    def __init__(self):
        super().__init__()
        self.arg_parser.add_argument("--class_thres", type=float, default=0.5)
        self.arg_parser.add_argument("--obj_thres", type=float, default=0)
        self.arg_parser.add_argument("--obj", type=str, choices=obj_names, default="person")     
        self.model_config = 'config/yolov3.cfg'
        self.weights = 'weights/yolov3.weights'
        
  
    def initialize(self, opt, gpu=None, prerained_model_name = 'vgg19_finetuned_all'):
        self.opt = opt
        
        self.device = config.device
        if gpu is not None:
            self.device = torch.device(f"cuda:{gpu}")
            
        print("Loading YOLOv3 weights...")
        self.net = cv2.dnn.readNetFromDarknet(self.model_config, self.weights)
        print("Done...")
        
        print("Loading sentiment analysis weights...")
        self.model = AlexNet if 'hybrid' in prerained_model_name else VGG19
        self.model = self.model('weights/{}.pth'.format(prerained_model_name)).to('cuda')
        self.model.eval()
        print("Done...")
        
        
    def get_img_size(self):
        return (416, 416)
      
    def get_scores(self, imgs, visualize=False):
        
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    
        (W, H) = (None, None)
    
        scores = [0 for i in range(len(imgs))]
        record = {}
        crop_idx = 0
        frame_idx = 0
        cropped_imgs = []
        visual_imgs = []

       
        for frame in tqdm(imgs):
            if W is None or H is None:
                (H, W) = frame.shape[:2]
        
            # create a blob from frame and forward it to YOLO.
            # it will give us the bounding boxes and probabilities
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(ln)
 
       
            for output in outputs:

                for detected in output:

                    # get class ID and probability.
                    probability = detected[5:]
                    class_ID = np.argmax(probability)
                    confidence = probability[class_ID]

                    # check if probability higher than confidence, class_ID 0 correspond to "person"
                    if confidence > self.opt.class_thres and class_ID==0:
                      
                        # scaling the bounding boxes respect to frame size
                        # YOLO returns the bounding box parameters in the order of :
                        # center X , center Y, width , height
                        box = detected[0:4] * np.array([W, H, W, H])
                        (center_x, center_y, width, height) = box.astype("int")

                        # find top and left corner of bounding box
                    
                        x1 = int(center_x - (width / 2))

                        y1 = int(center_y - (height / 2))
                    
                        x2 = x1 + width
                   
                        y2 = y1 + height
                    
                        # select and crop the box area 
                        image = frame[y1:y2, x1:x2]
                        cropped_imgs.append(image)
                    
                        record[crop_idx]= frame_idx
                        
                        # draw rectangle in the picture 
                        # Here may results in overflow 
                        # if visualize:
                            #visual_img = np.copy(frame)
                            #cv2.rectangle(visual_img, (x1, y1), (x2, y2), (0, 255, 0), 2)   
                            #visual_imgs.append(visual_img)
                            # add to list
                    
                    
                        #boxes.append([x, y, int(width), int(height)])
                        #confs.append(float(confidence))
                        crop_idx += 1

            # applying non-maxima suppression
            #idxs = cv2.dnn.NMSBoxes(boxes, confs, args.confidence, args.nms)            
    
            frame_idx += 1 

        crop_scores = sentiment_analysis(cropped_imgs, self.model)
    
        for i, crop_score in enumerate(crop_scores):
            scores[record[i]] += crop_score

        if visualize:
            return scores, visual_imgs
        else:
            return scores

       
