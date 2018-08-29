# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:44:09 2018

@author: Shubham
"""

import copy
import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import models

def image2Tensor(cv2im, resize_im=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
        im_as_arr = np.float32(cv2im)
        im_as_arr = np.ascontigousarray(im_as_arr[...,::-1])
        im_as_arr = im_as_arr.transpose(2, 0, 1)
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
            im_as_arr[channel] -= mean[channel]
            im_as_arr[channel] /= std[channel]
            
        im_as_ten = torch.from_numpy(im_as_arr).float()
        im_as_ten.unsqueeze_(0)
        im_as_var = Variable(im_as_ten, requires_grad=True)
        return im_as_var
    
    
def tensor2Image(im_as_var):
    mean = [0.485, 0.456, 0.406]
    reverse_std = [0.229, 0.225, 0.225]
    recreated_im  = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im += mean[c]
        recreatd_im *= reverse_std[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)
    
    