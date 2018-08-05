# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:49:43 2018
Extract text features using Bidirectional features
@author: Shubham
"""

import numpy as np
import torch
import sys
import argparse
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import os, sys
from loadGlove import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input parameters
input_word_vector_len = 300
hidden_size = 128
num_layers = 2
num_classes = 10
lr = 1e-4

# load glove embedding        
glove2word2vec(glove_input_file=r"C:\Users\Shubham\Desktop\glove.42B.300d.txt", 
                word2vec_output_file="gensim_glove_vectors.txt")

glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)

class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        out, hidden = self.lstm(x, (c0, h0))
        
        out = self.fc(out[:, -1, :])
        return out
  
# sentence global vector encoder
class Text_Encoder:
    def __init__(self, model, embedding, sentence):
        self.embedding = embedding
        self.sentence = sentence
        self.model = model
        
    def word_encode(self):
        M = TextFeatures.form_feature_matrix(self.embedding, self.sentence)
        return self.model(M)