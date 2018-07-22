# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 11:09:44 2018

@author: Shubham
"""

import numpy as np
import os
import sys

def loadGlove(filename):
    print("Loading Glove Embeddings...")
    f = open(filename, 'r', encoding='utf-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Extraction Complete...")
    return model
        
