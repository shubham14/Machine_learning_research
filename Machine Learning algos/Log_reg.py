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
    # initialize the data
    def __init__(self, x_train, y_train, x_test, y_test, num_epochs,
                 lr):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_epochs = num_epochs
        self.lr = lr
    
    