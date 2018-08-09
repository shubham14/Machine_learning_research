# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:33:30 2018

@author: Shubham
"""

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn import preprocessing
from PIL import Image 
import argparse
import re
import cv2
from collections import defaultdict

# parameters for the training and validation set
BATCH_SIZE = 20
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
TRAINING_DIR = r'C:\Users\Shubham\Desktop\data\tiny-imagenet-200\train'

NUM_VAL_IMAGES = 9832
VAL_IMAGES_DIR = r'C:\Users\Shubham\Desktop\data\tiny-imagenet-200\val'
IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS

BASE_DIR = r'C:\Users\Shubham\Desktop\data\tiny-imagenet-200'
LABEL_FILE = r'C:\Users\Shubham\Desktop\data\tiny-imagenet-200\words.txt'

# class for loading the data into numpy array 
class Data_Loader:
    
    def __init__(self, batch_size, num_classes, num_image_per_class):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_image_per_class = num_image_per_class
       
    
    def hasNumbers(self, inputString):
        return bool(re.search(r'\d', inputString))
        
    # directory depends on whether the loading the testing, validation or the 
    # testing phase, loading the images into a numpy array
    def load_to_array(self, directory):
        data = np.zeros((self.num_classes, self.num_image_per_class,
                                 IMAGE_SIZE, IMAGE_SIZE))
        class_files = os.listdir(BASE_DIR + '\\' + directory)
        final_image_file_name = list(map(lambda x: TRAINING_DIR + '\\' + x + 
                                         '\\' + 'images', class_files))
        for i, file in enumerate(final_image_file_name):
            images = os.listdir(file)
            for j, img in enumerate(images):
                img_path = file + '\\' + img 
                print (img_path)
                d = cv2.imread(img)
                b1, g1, r1 = cv2.split(d)
                d = cv2.merge([r1, g1, b1])
                data[i][j] = d
        return data
    
    #loading the labels as targets into an array
    def label_to_array(self):
        label_txt = open(LABEL_FILE, 'r+')
        data = label_txt.read()
        label_dict = defaultdict(list)
        r = ' '.join(' '.join(data.split('\n')).split('\t')).split(' ')
        for i, ele in enumerate(r):
            if self.hasNumbers(ele):
                key = ele
            else:
                label_dict[key].append(ele)
        return label_dict