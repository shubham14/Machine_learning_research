# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:42:11 2018

@author: Shubham
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

input_tensor = Input(shape=(224, 224, 3))
model = InceptionV3(input_tensor=input_tensor, weights='imagenet',
                    include_top=True)