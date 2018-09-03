# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 20:45:09 2018

@author: Shubham
"""

import numpy as np
import pandas as pd
import math
from utils import *

class MultinomialRegression:
    # initialize the data and training parameters
    # K is the number of classes
    def __init__(self, x_train, y_train, x_test, y_test, num_epochs,
                 lr, K):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_epochs = num_epochs
        self.lr = lr
        self.K = K
        
    # initialize the features \theta
    def init_features(self):
        m, _ = np.shape(self.x_train)
        theta = np.random.random((m, 1))
        return theta
        
    # log likelihood function, 
    def loglikelihood(self, y):
        p = sigmoid(y)
        l = 0
        m, _ = np.shape(self.x_train)
        for i in range(self.K):
            for j in range(m):
                l += indicator(self.y_train[j], i) * np.log(softmax(self.train[j], self.K))
        return l
    
    
    # optimize the weights 
    def train(self):
        theta = self.init_features()
        for _ in self.num_epochs:
            cost_diff = np.dot(self.y_train - sigmoid(np.dot(self.x, theta)) * self.x_train)
            theta += self.lr * cost_diff
            return theta