# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 13:27:10 2018

@author: Shubham
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
import torchtext
from nltk import word_tokenize
from torch import optim
import time
import numpy as np
import random
from config import config

class DataPrep:
    def __init__(self, max_len, train_path, test_path):
        self.max_len = max_len
        self.embed_path = r'C:\Users\Shubham\Desktop\data\quora\glove.840B.300d\glove.840B.300d.txt'
        
    def process_data(self):
        text = torchtext.data.Field(lower=True, batch_first=True, 
                                    tokenize=word_tokenize, fix_length=self.max_len)
        target = torchtext.data.Field(sequential=False, use_vocab=False, 
                                      is_target=True)
        train = torchtext.data.TabularDataset(path=train_path, format='csv',
                                               fields={'question_text': ('text',text),
                                              'target': ('target',target)})
        t1 = text.build_vocab(train, min_freq=1)
        t = text.vocab.load_vectors(torchtext.vocab.Vectors(self.embed_path))
        random_state = random.getstate()
        train, val = train.split(split_ratio=0.8, random_state=random_state)
        batch_size = 512
        train_iter = torchtext.data.BucketIterator(dataset=train,
                                           batch_size=batch_size,
                                           sort_key=lambda x: x.text.__len__(),
                                           shuffle=True,
                                           sort=False)

        val_iter = torchtext.data.BucketIterator(dataset=val,
                                         batch_size=batch_size,
                                         sort_key=lambda x: x.text.__len__(),
                                         train=False,
                                         sort=False)
        return t1, t, train_iter, val_iter
    
    
    
class LSTM(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, hidden_dim=128, static=True):
        super(LSTM, self).__init__()
        self.name = 'LSTM'
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=0.5)
        self.emndding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            dropout=0.5)
        self.hidden2label = nn.Linear(hidden_dim*2, 1)
        
    def forward(self, sents):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        y = self.hidden2label(self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1)))
        return y
 
    
class AttentionHelper(nn.Module):
    def __init__(self, embed_vectors, input_dim, dropout_p, embedding_size, hidden_size, output_size):
        super(AttentionHelper, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding.from_pretrained(embed_vectors)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=self.input_dim,
                          hidden_size=self.hidden_size,
                          bidirectional=self.bidirectional,
                          batch_size=True)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.Whc = nn.Linear(hidden_size * 2, hidden_size)
        self.Ws = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        gru_out, hidden = self.gru(embedded, hidden)
        attn_prod = torch.mm(self.attn(hidden)[0], encoder_outputs.t())
        attn_weights = F.softmax(attn_prod, dim=1)
        context = torch.mm(attn_weights, encoder_outputs)
        
        hc = torch.cat([hidden[0], context], dim=1)
        out_hc = F.tanh(self.Whc(hc))
        output = F.log_softmax(self.Ws(out_hc), dim=1)
        
        return output, hidden, attn_weights
        
        
class HierarchialAttentionNetwork(nn.Module):
    def __init__(self, embed_vectors, embedding_dim, hidden_dim):
        super(HierarchialAttentionNetwork, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embed_vectors)
        
        self.max_len = config.sentence_max_size
        self.input_dim = config.word_embedding_dimension
        self.hidden_dim = 50
        self.bidirectional = config.bidirectional
        self.drop_out_rate = config.drop_out
        self.context_vector_size = [100, 1]
        self.out_label_size = 2

        # dropout layer
        self.drop = nn.Dropout(p=self.drop_out_rate)

        # word level
        self.word_GRU = nn.GRU(input_size=self.input_dim,
                               hidden_size=self.hidden_dim,
                               bidirectional=self.bidirectional,
                               batch_first=True)
        self.w_proj = nn.Linear(in_features=2*self.hidden_dim if self.bidirectional else self.hidden_dim,
                                out_features=2*self.hidden_dim)
        self.w_context_vector = nn.Parameter(torch.randn(self.context_vector_size).float())
        self.softmax = nn.Softmax(dim=1)

        # sentence level
        self.sent_GRU = nn.GRU(input_size=2*self.hidden_dim if self.bidirectional else self.hidden_dim,
                               hidden_size=self.hidden_dim,
                               bidirectional=self.bidirectional,
                               batch_first=True)
        self.s_proj = nn.Linear(in_features=2*self.hidden_dim if self.bidirectional else self.hidden_dim,
                                out_features=2*self.hidden_dim)
        self.s_context_vector = nn.Parameter(torch.randn(self.context_vector_size).float())

        # document level
        self.doc_linear1 = nn.Linear(in_features=2*self.hidden_dim if self.bidirectional else self.hidden_dim,
                                     out_features=2*self.hidden_dim)
        self.doc_linear2 = nn.Linear(in_features=2*self.hidden_dim,
                                     out_features=self.hidden_dim)
        self.doc_linear_out = nn.Linear(in_features=self.hidden_dim,
                                        out_features=self.out_label_size)
        
    def forward(self, x, sent_num):
        x, _ = self.word_GRU(x)
        Hw = F.tanh(self.w_proj(x))
        w_score = self.softmax(Hw.matmul(self.w_context_vector))
        x = x.mul(w_score)
        x = torch.sum(x, dim=1)
        
        x = _align_sent(x, sent_num=sent_num).cuda()
        x, _ = self.sent_GRU(x)
        Hs = F.tanh(self.s_proj(x))
        s_score = self.softmax(Hs.matmul(self.s_context_vector))
        x = x.mul(s_score)
        x = torch.sum(x, dim=1)
        x = F.sigmoid(self.doc_linear1(x))
        x = F.sigmoid(self.doc_linear2(x))
        x = self.doc_linear_out(x)
        return x
    
    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels
        
        
def _align_sent(input_matrix, sent_num, sent_max=None):
    embedding_size = input_matrix.shape[-1]      # To get the embedding size of the sentence
    passage_num = len(sent_num)                  # To get the number of the sentences
    if sent_max is not None:
        max_len = sent_max
    else:
        max_len = torch.max(sent_num)
    new_matrix = autograd.Variable(torch.zeros(passage_num, max_len, embedding_size))
    init_index = 0
    for index, length in enumerate(sent_num):
        end_index = init_index + length

        # temp_matrix
        temp_matrix = input_matrix[init_index:end_index, :]      # To get one passage sentence embedding.
        if temp_matrix.shape[0] > max_len:
            temp_matrix = temp_matrix[:max_len]
        new_matrix[index, -length:, :] = temp_matrix

        # update the init_index of the input matrix
        init_index = length
    return new_matrix
        
if __name__ == "__main__":
    train_path = r'C:\Users\Shubham\Desktop\data\quora\train.csv'
    test_path =  r'C:\Users\Shubham\Desktop\data\quora\test.csv'
    data = DataPrep(50, train_path, test_path)
    print("Starting")
    st = time.time()
    t, t1, train_iter, val_iter = data.process_data()
    end = time.time()
    t_time = end - st
    print("ending: %s seconds" %t_time)   
#    att = AttentionHelper(100)
#    x = Variable(torch.randn(16,30,100))
#    y = att.forward(x)