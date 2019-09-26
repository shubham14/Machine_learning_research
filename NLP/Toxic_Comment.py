'''
Data Exploration for toxic comment classification
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from os.path import join as pjoin
from helpers import clean, mode
import pandas as pd
import re
from gensim.models import Word2Vec

BASE_DIR = r'C:\\Users\\Shubham\\Desktop\\Flow-Gans'
TRAIN_DIR = pjoin(BASE_DIR, 'train.csv')
TEST_DIR = pjoin(BASE_DIR, 'test.csv')

@mode("train")
def proc_data(dir_path):
    df = pd.read_csv(dir_path)
    X = df["comment_text"].fillna("unknown").values
    if "train" in dir_path:
        y = df[["toxic", "severe_toxic", "obscene", "threat", "insult", 
                "identity_hate"]].values
        return X, y
    return X

def tokenize(X):
    corpus_text = '\n'.join(df[:50000]['comment_text'])
    sentences = corpus_text.split('\n')
    sentences = [line.lower().split(' ') for line in sentences]
    sentences = [clean(s) for s in sentences if len(s) > 0]

train = pd.read_csv(TRAIN_DIR)
test = pd.read_csv(TEST_DIR)
