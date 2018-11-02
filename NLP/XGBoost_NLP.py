pip # -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 01:07:31 2018

@author: Shubham
"""

import sklearn
from sklearn import datasets
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class data:
    def datsets(self):
        train = datasets.fetch_20newsgroups(subset='train', remove=('headers',
                                                                    'footers',
                                                                    'quotes'))
        test = datasets.fetch_20newsgroups(subset='test', remove=('headers',
                                                                    'footers',
                                                                    'quotes'))
        X, y = train.data, train.target
        X_test_docs, y_test = test.data, test.target
        X_train_docs, X_val_docs, y_train_docs, y_val_docs = train_test_split(
                X, y, test_size=0.2, random_state=42)
        tfidf = TfidfVectorizer(min_df=0.005, max_df=0.5).fit(X_train_docs)
        
        # remove sparsity
        X_train_tfidf = tfidf.transform(X_train_docs).tocsc()
        X_val_tfidf = tfidf.transform(X_val_docs).tocsc()
        X_test_tfidf = tfidf.transform(X_test_docs).tocsc()
        
        return X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train_docs, y_val_docs, y_test
    
class model:
    def __init__(self, X_train, X_test, y_train, t_test):
        self.clf = xgb.XGBClassifier(n_estimators=50000)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def fit1(self):
        self.clf = self.clf.fit(self.X_train, self.y_train, 
                                eval_set=[(self.X_train, self.y_train), 
                                          (self.X_val, self.y_val)],
                                verbose=100,
                                early_stopping_rounds=10)
        
    def predict(self):
        y_pred = self.clf.predict(self.X_test)
        return y_pred