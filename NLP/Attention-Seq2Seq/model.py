# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 23:42:23 2018

@author: Shubham
"""

import torch
from torch.nn import nn
from torch.autograd import Variable 
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1,
                 dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, 
                          dropout=self.dropout, bidirectional=True)
    
    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)
        outputs, hidden = self.gru(packed, hidden)
        outputs, outputs_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size]
        return outputs, hidden
    

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, n_layers=1,
                 dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True)
        
    def forward(self, input_seq, input_len, hidden=None):
        batch_size = input_seqs.size(1)
        embedding = self.embedding(input_seq)
        embedding = embedding.transpose(0, 1)
        sort_idx = np.argsort(-input_len)
                