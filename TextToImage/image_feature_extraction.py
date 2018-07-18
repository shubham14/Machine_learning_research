# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:42:11 2018
Extract image features with pretrained inception version3 network
@author: Shubham
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse

model_conv = torchvision.models.inception_v3(pretrained='imagenet')

freeze_layers = True

if freeze_layers:
    for i, param in model_conv.named_parameters():
        param.requires_grad = False
        
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, n_class)

input_shape = 299
batch_size = 32
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
scale = 360
use_parallel = True
use_gpu = False
epochs = 100

# change this accordingly
dim_vector = 100

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(scale),
        transforms.RandomResizedCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
        
    'val': transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),}
    
# write data loaders and reshape accordingly
    
# extract word features from the mixed 6e layer
class InceptionNetWord(nn.Module):
    def __init__(self, original_model):
        super(InceptionNetword, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-9])
        self.W = nn.Linear(dim_vector, 768)
        
    def forward(self, x):
        x = self.features(x)
        x = self.W(x)
        return x
    
# extract global sentence features from the mixed 6e layer
class InceptionNetSent(nn.Module):
    def __init__(self, original_model):
        super(InceptionNetSent, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-9])
        self.W_prime = nn.Linear(dim_vector, 2048)
        self.avg_pool = nn.AvgPool2d()
        
    def forward(self, x):
        x = self.features(x)
        x = self.W_prime(x)
        x = avg_pool(x)
        return x
    