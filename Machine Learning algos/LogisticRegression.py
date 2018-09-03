# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 12:42:21 2018

@author: Shubham
"""

import numpy as np
import pandas as pd
import math
from utils import *

class LogisticRegression:
    # initialize the data and training parameter
    def __init__(self, x_train, y_train, x_test, y_test, num_epochs,
                 lr):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_epochs = num_epochs
        self.lr = lr
       
    # initialize the features \theta
    def init_features(self):
        m, _ = np.shape(self.x_train)
        theta = np.random.random((m, 1))
        return theta
        
    def loglikelihood(self, y):
        p = sigmoid(y)
        l = -y * np.log(p) + (1 - y) * np.log(1 - p)
        return l
    
    # optimize the weights 
    def train(self):
        theta = self.init_features()
        for _ in self.num_epochs:
            cost_diff = np.dot(self.y_train - sigmoid(np.dot(self.x, theta)) * self.x_train)
            theta += self.lr * cost_diff
            return theta
    
    