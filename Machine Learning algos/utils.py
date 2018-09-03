# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 12:45:46 2018
Contains common utility functions for the Machine Learning algorithms
@author: Shubham
"""

import numpy as np
    
def sigmoid(x):
    return 1./1 + np.exp(-int(x))

# K is the number of classes
def softmax(x, prob_classes):
    num = np.exp(x)
    norm = num/np.sum(np.exp(x), axis=0)
    return norm

def indicator(a, b):
    if a == b:
        return 1
    return 0