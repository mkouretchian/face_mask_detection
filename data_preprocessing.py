#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:29:18 2021

@author: roji
"""

import glob
from PIL import Image
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import random

random.seed(10)

directory_images = '/Users/roji/Documents/face_mask_detection_data/images'
directory_annotations = '/Users/roji/Documents/face_mask_detection_data/annotations'

list_of_files_png = glob.glob(directory_images+str('/*.png'))
list_of_files_xml = glob.glob(directory_annotations+str('/*.xml'))
list_of_files_jpg = []
data_size = len(list_of_files_xml)

train_size = int(0.9*data_size)




for item in list_of_files_png:
    img = Image.open(item)
    rgb_im = img.convert('RGB')
    save_location = item[:-4]+str('.jpg')
    list_of_files_jpg.append(save_location)
    rgb_im.save(save_location)
    
set_index = np.arange(len(list_of_files_xml))
random.shuffle(set_index)
train_index = set_index[:train_size]
validation_index = set_index[train_size:]  
# print(validation_index)  
# print(len(list_of_files_jpg))
train_list_jpg = [list_of_files_jpg[i] for i in train_index]
train_list_xml = [list_of_files_xml[i] for i in train_index]
validation_list_jpg=[list_of_files_jpg[j] for j in validation_index]
validation_list_xml = [list_of_files_xml[j]  for j in validation_index]


    
    
def get_objects(xml_file):
    objects = [] 
    df = pd.DataFrame()
    annotation = ET.parse(xml_file)
    root = annotation.getroot()
    size = root.find('size')
    file_name = root.find('filename').text
    dimensions = (int(size.find('width').text),
                              int(size.find('height').text),
                              int(size.find('depth').text))
    for obj in root.findall('object'):
        new_object = {'name':obj.find('name').text}
        bbox_tree = obj.find('bndbox')
        new_object['bbox'] = (int(bbox_tree.find('xmin').text), 
                              int(bbox_tree.find('ymin').text), 
                              int(bbox_tree.find('xmax').text), 
                              int(bbox_tree.find('ymax').text),
                              )
        
        objects.append(new_object)
    df = pd.DataFrame(objects)
    return df , dimensions ,file_name
        
def change_labels_to_numerics(aux):
    if aux == 'with_mask':
        return 0
    elif aux == 'without_mask':
        return 1
    else :
        return 2

def change_format_bbox(df,dimensions):
    
    width , height,_ = dimensions
    x_center_arr = []
    y_center_arr = []
    height_df_arr = []
    width_df_arr = []
    
    
    for index , row in df.iterrows():
        x_min , y_min, x_max, y_max = row['bbox']
        x_center = (x_min + x_max)/(2*width)
        y_center = (y_min + y_max)/(2*height)
        height_df = (y_max - y_min)/(height)
        width_df = (x_max - x_min)/(width)
        x_center_arr.append(x_center)
        y_center_arr.append(y_center)
        height_df_arr.append(height_df)
        width_df_arr.append(width_df)
    df = df.assign(x_center = x_center_arr)
    df = df.assign(y_center = y_center_arr)
    df = df.assign(height_df = height_df_arr)
    df = df.assign(width_df = width_df_arr)
        
    return df




def train_validation_image(train_valid_list_jpg,category):
    
    for item in train_valid_list_jpg:
        print(len(train_valid_list_jpg))
        if category == 'train':
            directory = '/Users/roji/Documents/face_mask_detection_data/train_image/'
        else:
            directory = '/Users/roji/Documents/face_mask_detection_data/validation_image/'
        im1 = Image.open(item)
        im2 = im1.copy()
        fname = item.split("/")[-1]
        directory = directory+str(fname)
        im2 = im2.save(directory)
  

      
def train_validation_annotation(train_valid_annotation,category) : 
    
        
    for item in train_valid_annotation:
        
        if category == 'train':
            directory = '/Users/roji/Documents/face_mask_detection_data/train_annotations/'
        else :
            directory = '/Users/roji/Documents/face_mask_detection_data/validation_annotations/'
        
        df , dimensions , filename = get_objects(item)
        
        df['name'] = df['name'].apply(lambda x : change_labels_to_numerics(x))
        
        df = change_format_bbox(df, dimensions)
        
        df_new = df[['name','x_center','y_center','width_df','height_df']]
        
        filename = filename[:-4]+str('.txt')
        
        fname = directory + str(filename)
        
        np.savetxt(fname, df_new.values, delimiter = '\t')
        
        

train_validation_image(train_list_jpg, category='train')

train_validation_image(validation_list_jpg, category = 'validation')

train_validation_annotation(train_list_xml, category = 'train')

train_validation_annotation(validation_list_xml, category = 'validation')
        
        
        
        
        


        
    
