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

# parameters 
device = torch.device('cpu')
filename = r'C:\Users\Shubham\Desktop\data\tiny-imagenet-200\train'
mean = [0.76487684, 0.75205952, 0.74630833]
std = [0.27936298, 0.27850413, 0.28387958]

# network parameters
num_epochs = 20
lr = 1e-3

# load images
data_load = DataLoader(filename, True, mean, std)
data =data_load.load()

# Network definition for feature extraction
# 200 classes for tiny imagenet
class Network(nn.Module):
    def __init__(self, num_classes=200):
        super(Network, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True)
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True)
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, num_classes)
        self.final_layer = nn.Sigmoid(num_classes)
        
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.final_layer(out)
        return out
    
#defining the device to run the network
model = Network(10).to(device)

# optimizer and parameters defined
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training loop 
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(zip(data.keys(), data.values())):
        inp, labels = data
        optimizer.zero_grad()
        outputs = model(inp)
        loss = criterion(labels, outputs)
        loss.backward()
        optimizer.step()
        
        running_step += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            
print ('Finished Training')