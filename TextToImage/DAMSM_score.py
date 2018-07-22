# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:02:39 2018

Deep Attentional Multimodal Similarity Model score based on text and image features
@author: Shubham
"""
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import sys
from image_feature_extraction import *
from text_feature_extraction import *

model_conv = torchvision.models.inception_v3(pretrained='imagenet')

# GloVe embeddings
glove = open('C:\Users\Shubham\Desktop\glove.42B.300d.txt', 'r')
embed = glove.read()
embed = embed.split()
embed = map(lambda x: float(x), embed)


# instantiate the Word and Image extraction features
enc = EncoderRNN(vocab_size, max_len, hidden_size, input_dropout_p=0,
                 dropout_p=0, n_layers=1, bidirctional=True,
                 rnn_cell='lstm', variable_lengths=False, embedding=None, 
                 update_embedding=True)


e, h, e_prime = enc(inputs)
v = InceptionNetWord(model_conv)
v_prime = InceptionNetSent(model_conv)

e_transpose = e.permute(1,0)
s = torch.mul(e_transpose, v)

s_exp = torch.exp(s)
sum_vector = torch.sum(s_exp, dim=0)
c = s_exp.permute(1, 0) / sum_vector
c = c.permute(1, 0)

