#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:13:27 2021

@author: roji
"""


import torch
import time
from PIL import Image
import gluoncv as gcv
import mxnet as mx
import cv2
import wandb





img = Image.open('/Users/roji/Documents/face_mask_detection_data/train_image/maksssksksss5.jpg')
model= torch.hub.load('ultralytics/yolov3','custom',path = 'best.pt')
output = model([img])
print(output.pandas().xyxy[0])

# model= torch.hub.load('ultralytics/yolov3','custom',path = 'best.pt')
cap = cv2.VideoCapture(0)
# time.sleep(1)
# axes = None
# num_frames = 60

# for i in range(num_frames):
#     ret , frame = cap.read()
#     frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

display_ids = {'with_mask': 0 ,'without_mask':1, 'mask_weared_incorrectly':2}

class_id_to_label = {int(v) : k for k,v in display_ids.items()}

def bounding_boxes(v_box,v_labels,v_scores):
    
    _ ,raw_image = cap.read()
    all_boxes = []
    output = model(raw_image)
    data_frame = output.pandas().xyxy[0]
    for b_i , box in enumerate(data_frame):
        box_data = {
            "position" : {
                'minX' : box.xmin,
                'maxX' : box.xmax,
                'minY' : box.ymin,
                'maxY' : box.ymax
                },
            'class_id' : display_ids[data_frame.name[b_i]],
            'box_caption' : "%s (%.3f)" %(data_frame.name[b_i],data_frame.confidence[b_i]),
            'domain' : "pixel",
            'scores' : {"score":data_frame.confidence[b_i]}
            
            }
        all_boxes.append(box_data)
    box_image = wandb.Image(raw_image,boxes = {'predictions':{"box_data" : all_boxes , "class_labels":class_id_to_label}})
    return box_image