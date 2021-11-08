#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:34:18 2021

@author: roji
"""
import os
print(os.getcwd())
directory = '/Users/roji/Documents/face_mask_detection/data.yaml'

with open(directory,'w') as f :
    f.write('train: ')
    f.write('/Users/roji/Documents/face_mask_detection/train_image')
    f.write('\n')
    f.write('val: ')
    f.write('/Users/roji/Documents/face_mask_detection/validation_image')
    f.write('\n')
    f.write('nc: ')
    f.write(' 3')
    f.write('\n')
    f.write('names: ')
    f.write(" ['with_mask', 'without_mask', 'mask_weared_incorrectly']")