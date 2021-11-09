#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:13:27 2021

@author: roji
"""


import torch
import numpy as np
import time
from PIL import Image
import gluoncv as gcv
import mxnet as mx
import cv2
import wandb
from matplotlib import pyplot as plt



# wandb.init(project = 'test',entity = "kouretchian")

# img = plt.imread('/Users/roji/Documents/face_mask_detection_data/train_image/maksssksksss5.jpg')
# model= torch.hub.load('ultralytics/yolov3','custom',path = 'best.pt')
# output = model([img])
# print(output.pandas().xyxy[0])

# # model= torch.hub.load('ultralytics/yolov3','custom',path = 'best.pt')
# #cap = cv2.VideoCapture(0)
# # time.sleep(1)
# # axes = None
# # num_frames = 60

# # for i in range(num_frames):
# #     ret , frame = cap.read()
# #     frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

# display_ids = {'with_mask': 0 ,'without_mask':1, 'mask_weared_incorrectly':2}

# class_id_to_label = {int(v) : k for k,v in display_ids.items()}

# while True:
#     raw_image = plt.imread('/Users/roji/Documents/face_mask_detection_data/train_image/maksssksksss5.jpg')
#     # _ ,raw_image = cap.read()
#     all_boxes = []
#     output = model(raw_image)
#     data_frame = output.pandas().xyxy[0]
#     for b_i , box in data_frame.iterrows():
#         box_data = {
#             "position" : {
#                 'minX' : box.xmin,
#                 'maxX' : box.xmax,
#                 'minY' : box.ymin,
#                 'maxY' : box.ymax
#                 },
#             'class_id' : display_ids[data_frame.name[b_i]],
#             'box_caption' : "%s (%.3f)" %(data_frame.name[b_i],data_frame.confidence[b_i]),
#             'domain' : "pixel",
#             'scores' : {"score":data_frame.confidence[b_i]}
            
#             }
#         all_boxes.append(box_data)
#     box_image = wandb.Image(raw_image,boxes = {'predictions':{"box_data" : all_boxes , "class_labels":class_id_to_label}})
    
#     wandb.log({"img":[box_image]})
#     break
model= torch.hub.load('ultralytics/yolov3','custom',path = 'best.pt')
cap = cv2.VideoCapture(0)
time.sleep(1)
colors = np.random.uniform(0,255,size = (3,3))
classes = {0:colors[0],1:colors[1],2:colors[2]}
classes_name = {0:"with_mask",1:"without_mask",2:"mask_weared_incorrectly"}
while True:

    _ , raw_image = cap.read()
    output = model(raw_image)
    data_frame = output.pandas().xyxy[0]
    print(data_frame)
    for i , row in data_frame.iterrows():
        confidence = row.confidence
        idx = row["class"]
        (startX , startY , endX, endY) = int(row.xmin) , int(row.ymin) , int(row.xmax) , int(row.ymax)
        label = "{}: {:.2f}%".format(classes_name[idx], row.confidence*100)
        print("[info] {}".format(label))
        cv2.rectangle(raw_image,(startX,startY),(endX,endY),classes[idx],2)
        y = startY -15 if startY -15 > 15 else startY + 15
        cv2.putText(raw_image,label,(startX , y),cv2.FONT_HERSHEY_SIMPLEX,0.5,classes[idx],2)
    cv2.imshow("Output",raw_image)
    cv2.waitKey()