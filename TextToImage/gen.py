# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:01:00 2018
See what this GLU is
@author: Shubham
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.utils.model_zoo as model_zoo
import argparse
from PIL import Image
from torch.autograd import Variable
from loadGlove import *

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
    
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

def conv3x3(in_dim, out_dim):
    return nn.Conv2d(in_dim, out_dim, kernel_size=3, 
                     stride=1, padding='same', bias=False)

def upBlock(in_dim, out_dim):
    block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_dim, out_dim * 2),
            nn.BatchNorm2d(out_dim * 2),
            GLU())
    return block


class CA_Net(nn.Module):
    def __init__(self, c_dim):
        super(CA_NET, self).__init__()
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()
        
    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim]
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparameterize(mu, logvar)
        return c_code, mu, logvar
    
    

class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super.gf_dim = ngf
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf
        
        self.define_module()
        
    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
                nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
                nn.BatchNorm1d(ngf * 4 * 4 * 2),
                GLU())
        self.upsample1 = upBlock(ngf, ngf//2)
        self.upsample2 = upBlock(ngf//2, ngf//4)
        self.upsample3 = upBlock(ngf//4, ngf//8)
        self.upsample4 = upBlock(ngf//8, ngf//16)
        
    def forward(self, z_code, c_code):
        c_z_code = torch.cat((c_code, z_code), 1)
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        return out_code
    