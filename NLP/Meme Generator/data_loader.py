# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 22:20:26 2018

@author: Shubham
"""

import numpy as np
import os
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import gensim
import argparse

# dataPath has to be changed according to systems 
dataPath = 'C:/Users/Shubham/Desktop/data/Embeddings/'

# parse inputs to the program
parser = argparse.ArgumentParser(description='Choose embedding type for text generation')
parser.add_argument('--embedding', help='choose embedding type, w: for word and c: for char')
args = parser.parse_args()

class DataLoader:
    def __init__(self, dataPath, args):
        self.dataPath = dataPath
        self.args = args
    
    def define_model(self):
        if self.args == 'word':
            model = gensim.models.KeyedVectors.load_word2vec_format(self.dataPath + 'GoogleNews-vectors-negative300.bin',
                                                                binary=True)
            return model
        else:
            vectors = {}
            file_path = self.dataPath + 'glove.840B.300d.txt'
            with open(file_path, 'rb') as f:
                for line in f:
                    line_split = line.strip().split(" ")
                    vec = np.array(line_split[1:], dtype=float)
                    word = line_split[0]
                    for char in word:
                        if ord(char) < 128:
                            if char in vectors:
                                vectors[char] = (vectors[char][0] + vec,
                                                 vectors[char][1] + 1)
                            else:
                                vectors[char] = (vec, 1)
            return vectors