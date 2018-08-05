# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 11:09:44 2018
incorporate text embedding and size dimensions
@author: Shubham
"""

import numpy as np
import os
import sys
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

class TextFeatures:
    def form_feature_matrix(self, model, sentence):
        M = []
        for i, word in enumerate(sentence):
            if i < 1:
                M = np.vstack([sentence[i]])
            M = np.vstack([M, word])
        M = M.T
        return M
            