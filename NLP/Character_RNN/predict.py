# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:38:14 2018

@author: Shubham
"""

import sys
from model import *
from data_prepper import *
from torch.autograd import Variable

rnn = torch.load('char-rnn.pt')
l = glob.glob('/Users/Shubham/Desktop/Machine_learning_research/data/Char_RNN/names/*.txt')

all_categories = []
for filename in l:
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    
# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(line, n_predictions=3):
    output = evaluate(Variable(lineToTensor(line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions

if __name__ == '__main__':
    predict(sys.argv[1])