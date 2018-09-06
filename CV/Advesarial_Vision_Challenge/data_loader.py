# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:33:30 2018
Data loader for tiny Imagenet, contains 200 classes
@author: Shubham Dash
"""

# import necessary libraries
import numpy as np
import os
from PIL import Image 
from collections import defaultdict

filename = r'C:\Users\Shubham\Desktop\data\tiny-imagenet-200\train'

class DataLoader:
    # normalize is a bool for the data normalization
    # False by default
    def __init__(self, path, normalize, mean, std):
        self.path = path
        self.data = defaultdict(list)
        self.normalize = False
        
        if not mean and not std:
            # hard coded channel sizes to be 3
            self.mean = [0.0] * 3
            self.std = [1.0] * 3
        else:
            self.mean = mean
            self.std = std
        
    # mean and std are ch X 1 vectors
    # ch is the number of channels
    def normalize(self, img):
        img_norm = []
        for ch in range(3):
            img_norm[ch] = np.array(list(map(lambda x: (x - self.mean[ch])/self.std[ch],
                    img[ch])))
        return img_norm
        
    # load function returns a dictionary of images according to classes
    # normalize the data, for network input
    def load(self):    
        # list of all separate image filename
        r = os.listdir(self.path)
        len_1  = len(r)
        r = list(map(lambda x: self.path + '\\' + x + '\\images\\' , r))
        len_2 = len(r[0])
             
        for i in range(len(r)):
            img = os.listdir(r[i])
            imgname =  list(map(lambda x: r[i]  + x, img))
            for ele in imgname:
                if self.normalize:
                    norm_img = self.normalize(Image.open(ele))
                    self.data[i].append(norm_img)
                else:
                    self.data[i].append(np.array(Image.open(ele)))    
        return self.data
        
    