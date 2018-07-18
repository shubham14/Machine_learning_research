'''
Python implemetation of extracting text features using Bi_LSTM
SDash Jul 18
'''

import torch
import torch.nn as nn
import os
import sys
import nltk

class BaseRNN(nn.Module):
    
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"
    
    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p, dropout_p,
                 n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN cell")
            
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
        
class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p=0,
                 dropout_p=0, n_layers=1, bidirctional=True,
                 rnn_cell='lstm', variable_lengths=False, embedding=None, 
                 update_embedding=True):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
             input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if embedding is not None:
            self.embedding_weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True,
                                 bidirectional=bidirectional, dropout=dropout_p)
        
    def forward(self, input_var, input_length=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pack_padded_sequence(output, batch_first=True)
        global_sent_vector = torch.cat([b(x) for b in hidden], 1)
        return output, hidden, global_sent_vector
    
                