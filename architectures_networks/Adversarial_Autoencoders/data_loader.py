from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class DataLoader:
    def __init__(self, batch_size, num_classes=10, use_cuda=False):
        '''
        num_classes : 10 by default for MNIST Dataset 
                      change this parameter for different dataset

        '''
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.num_classes = num_classes

    def loadTrainData(self):
        '''
        Helper function to load train datasets using custom parameters
        '''
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./data', train=True, download=True,
                            transform=transforms.ToTensor()),
                batch_size=self.batch_size, shuffle=True, **kwargs)
    
    def to_var(self, x, use_cuda=False):
        x = Variable(x)
        if use_cuda:
            x = x.cuda()
        return x

    def one_hot(self, labels, class_size, use_cuda=False):
        '''
        One-hot encode the class labels for class conditional AAE
        '''
        targets = torch.zeros(labels.size(0), class_size)
        for i, label in enumerate(labels):
            targets[i, label] = 1
        return self.to_var(targets, use_cuda)