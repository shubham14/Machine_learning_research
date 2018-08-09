# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 20:01:39 2018

@author: Shubham
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from data_loader import *

device = torch.device('cpu')

# load images and labels
images = Data_Loader.load_to_array('train')
labels = Data_Loader.label_to_array()

# Network definition for feature extraction
class Network(nn.Module):
    def __init__(self, num_classes=10):
        super(Network, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
    
model = Network(10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

