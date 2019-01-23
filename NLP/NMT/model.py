# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 01:34:33 2019

@author: Shubham
"""

# model defined according to paper "Compression of Neural
# Machine Translation Models via Pruning"

import torch
import torch.nn as nn
from torchtext import *
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from etl import *

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_dataset()

class NMTEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, n_layers=1):
        super(NMTEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers)
        self.n_layers = n_layers
    
    def forward(self, word_inputs, hidden):
        embedded = self.embedding(word_inputs).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
    
    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers,1, self.hidden_dim))
        hidden = hidden.cuda()
        return hidden

class NMTDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(NMTDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        embeds = self.embedding(input).view(1, 1, -1)
        output = embeds
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class NMTAttnDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, input_dim, dropout_p,
                 max_length=100000):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        self.embeds = nn.Embedding(output_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.softmax = nn.LogSoftmax(dim=1)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.attn = nn.Linear(2 * hidden_dim, max_length)
        self.attn_combine = nn.Linear(2 * hidden_dim, hidden_dim)
    
    def forward(self, input, hidden, encoder_outputs):
        embeds = self.embeds(input)
        embeds = self.dropout(embeds)
        attn_weights = F.softmax(self.attn(torch.cat(embeds[0], hidden[0], 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        attn_combine = torch.cat((embeds[0], attn_apllied[0]), 1)
        attn_combine = F.relu(attn_combine)
        output = attn_combine
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output = self.softmax(output[0])
        return output, attn_weights, hidden
    
class LSTM(nn.Module):
    def __init__(self, inp_size, out_size, hidden_size, vocab_size, embedding_size, weights):
        super(LSTM, self).__init__()
        '''
        inp_size: batch size of the input
        '''
        self.inp_size = inp_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_sentence, batch_size=None):
        inp = self.word_embeddings(input_sentence)
        inp = inp.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.inp_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, self.inp_size, self.hidden_size))
        else:
            h_0 = Variable(torch.zeros(1, batch_size, hidden_size))
            c_0 = Variable(torch.zeros(1, batch_size, hidden_size))
        output, (final_hidden_state, final_cell_state) = self.lstm(inp, (h_0, c_0))
        return self.label(final_hidden_state[-1])

def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        # clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

learning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300

model = LSTM(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
loss_fn = F.cross_entropy

for epoch in range(10):
    train_loss, train_acc = train_model(model, train_iter, epoch)