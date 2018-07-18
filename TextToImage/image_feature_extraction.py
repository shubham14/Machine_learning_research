# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:42:11 2018

@author: Shubham
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
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

if freeze_layers:
    for i, param in model_conv.named_parameters():
        param.requires_grad = False
        
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, n_class)