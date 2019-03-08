'''
Model architecture for Adversarial Autoencoders based on https://arxiv.org/pdf/1511.05644.pdf
'''

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
from data_loader import *
from config import *

class Encoder(nn.Module):
    def __init__(self, input_size, latent_size, class_size, enc_cfg):
        '''
        Calculates q(z|x) from the given data
        input_size : (1, 28, 28) for MNIST dataset
        enc_cfg : Class object of Encoder Config class
        '''
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.class_size = class_size
        self.hidden_dim = 51200
        self.enc_cfg = enc_cfg
        self.cnn1 = nn.Sequential(nn.Conv2d(self.enc_cfg.in_channels, self.enc_cfg.cnn_filters[0], 3),
                                nn.ReLU(),
                                nn.BatchNorm2d(self.enc_cfg.cnn_filters[0]))
        self.cnn2 = nn.Sequential(nn.Conv2d(self.enc_cfg.cnn_filters[0], self.enc_cfg.cnn_filters[1], 3),
                                nn.ReLU(),
                                nn.BatchNorm2d(self.enc_cfg.cnn_filters[1]))
        self.cnn3 = nn.Sequential(nn.Conv2d(self.enc_cfg.cnn_filters[1], self.enc_cfg.cnn_filters[2], 3),
                                nn.ReLU(),
                                nn.BatchNorm2d(self.enc_cfg.cnn_filters[2]))
        self.cnn4 = nn.Sequential(nn.Conv2d(self.enc_cfg.cnn_filters[2], self.enc_cfg.cnn_filters[3], 3),
                                nn.ReLU(),
                                nn.BatchNorm2d(self.enc_cfg.cnn_filters[3]))
        self.fc1 = nn.Sequential(nn.Linear(self.hidden_dim, self.latent_dim),
                                nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(self.hidden_dim, self.latent_dim),
                                nn.ReLU())
    
    def forward(self, x):
        '''
        forward pass for the encoder
        '''
        cnn1_out = self.cnn1(x)
        cnn2_out = self.cnn2(cnn1_out)
        cnn3_out = self.cnn3(cnn2_out)
        cnn4_out = self.cnn4(cnn3_out)
        cnn4_out = cnn4_out.view(cnn4_out.size(0), -1)
        mu = self.fc1(cnn4_out)
        logvar = self.fc2(cnn4_out)
        return mu, logvar

class Reshape(nn.Module):
    """
    An Reshape module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Reshape, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

class Decoder(nn.Module):
    def __init__(self, input_size, latent_size, class_size, dec_cfg):
        '''
        Calculates p(x|x) from the latent representation
        input size : latent_dim 
        dec_cfg : Class object of Encoder Config class
        '''
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.class_size = class_size
        self.hidden_dim = 51200
        self.enc_cfg = enc_cfg
        self.fc1 = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim),
                                nn.ReLU(),
                                Reshape())
        self.cnn1 = nn.Sequential(nn.ConvTranspose2d(self.dec_cfg.in_channels, self.dec_cfg.cnn_filters[0], 3),
                                nn.ReLU(),
                                nn.BatchNorm2d(self.dec_cfg.cnn_filters[0]))
        self.cnn2 = nn.Sequential(nn.ConvTranspose2d(self.dec_cfg.cnn_filters[0], self.dec_cfg.cnn_filters[1], 3),
                                nn.ReLU(),
                                nn.BatchNorm2d(self.dec_cfg.cnn_filters[1]))
        self.cnn3 = nn.Sequential(nn.ConvTranspose2d(self.dec_cfg.cnn_filters[1], self.dec_cfg.cnn_filters[2], 3),
                                nn.ReLU(),
                                nn.BatchNorm2d(self.dec_cfg.cnn_filters[2]))
        self.cnn4 = nn.Sequential(nn.ConvTranspose2d(self.dec_cfg.cnn_filters[2], self.dec_cfg.cnn_filters[3], 3),
                                nn.ReLU(),
                                nn.BatchNorm2d(self.dec_cfg.cnn_filters[3]))
    
    def forward(self, z):
        '''
        forward pass for the decoder
        Note : the output for fc_out is reshaped from custom layer
        '''
        fc_out = self.fc1(z)
        cnn1_out = self.cnn1(fc_out)
        cnn2_out = self.cnn2(cnn1_out)
        cnn3_out = self.cnn3(cnn2_out)
        cnn4_out = self.cnn4(cnn3_out)
        X_hat = cnn4_out.view(cnn4_out.size(0), -1)
        return X_hat
        