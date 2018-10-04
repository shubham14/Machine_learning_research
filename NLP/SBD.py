# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:36:03 2018

@author: Shubham
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys
import random
from scipy import sparse
from sklearn import preprocessing
import time


class SBD:
    def __init__(self, clf, tr, ts, dictionary):
        self.clf = clf
        self.train_data = tr
        self.test_data = ts
        self.dictionary = dictionary
        
        # function to form feature vectors for training 
    def form_features(self, ele, file):
        if file == 'train':
            data = self.train_data
        else:
            data = self.test_data
        end_punctuations = ['?', '!', '"', '%', ')', '}']
        start_punctuations = ['"', '{', '(']
        char_list = ['?', '!', ',', '%', '"', '{', '}', '(', ')', '$', '#', ' ', '']
        salutation = ["mr", "mrs", "dr"]

        # components of the feature vector
        feature_vec = []
        psplit = ele.split()
        L = psplit[1]
        psplit_next = data[i + 1].split()
        if len(psplit_next) > 2:
            R = psplit_next[1]
        else:
            R = ''
        L = L.strip().lower()
        R = R.strip().lower()
        L1 = L; R1 = R
        if L1 in salutation:
            L1 = L1 + '.'
        if R in salutation:
            R1 = R1 + '.'
        if L1 in self.dictionary.keys():
            L_enc = self.dictionary[L1]
        else:
            L_enc = np.zeros((len(self.dictionary.values())))
        if R1 in self.dictionary.keys():
            R_enc = self.dictionary[R1]
        else:
            R_enc = np.zeros((len(self.dictionary.values())))
        feature_vec = list(L_enc) + list(R_enc)
        if len(L1) <= 3:
            feature_vec.append(0)
        else:
            feature_vec.append(1)
        
        if L1 not in char_list and (ord(L1[0])>=97 and ord(L1[0])<=122):
            feature_vec.append(0)
        else:
            feature_vec.append(1)
    
            
        if R1 not in char_list and (ord(R1[0])>=97 and ord(R1[0])<=122):
            feature_vec.append(0)
        else:
            feature_vec.append(1)
    
        # custom features
        if L1 in end_punctuations:
            feature_vec.append(True)
        else:
            feature_vec.append(False)  

        # check if L is a salutation then it is not an EOS
        if R1 in start_punctuations:
            feature_vec.append(True)
        else:
            feature_vec.append(False) 
            
        if len(L1.split('.')) > 1:
            feature_vec.append(True)
        else:
            feature_vec.append(False)
        return np.array(feature_vec)
  
    # file is either test or train
    def prep(self, file):
        end_punctuations = ['?', '!', '"', '%', ')', '}']
        start_punctuations = ['"', '{', '(']
        char_list = ['?', '!', ',', '%', '"', '{', '}', '(', ')', '$', '#', ' ', '']
        salutation = ["mr", "mrs", "dr"]
        if file == 'train':
            data = self.train_data
        else:
            data = self.test_data
        count = 0
        for ele in data:
            if 'EOS' in ele:
                count += 1
                
        X = np.zeros((count, (2*len(self.dictionary.values()) + 6))); Y = []
#        X = np.zeros((count, 3)); Y = []
        
        c = 0
        for i, ele in enumerate(data):
            feature_vec = []
            if i == len(data) - 1 and 'EOS' in data[i]:
                psplit = ele.split()
                L = psplit[1]
                L1 = L; R1 = R
                if L1 in salutation or (L1[-1] == '.'):
                    L1 = L1 + '.'
                if L1 in self.dictionary.keys():
                    L_enc = self.dictionary[L1]
                else:
                    L_enc = np.zeros((len(self.dictionary.values())))
                R_enc = np.zeros((len(self.dictionary.values())))
                feature_vec = list(L_enc) + list(R_enc)
                if len(L1) <= 3:
                    feature_vec.append(0)
                else:
                    feature_vec.append(1)
                    
                if L1 not in char_list and (ord(L1[0])>=97 and ord(L1[0])<=122):
                    feature_vec.append(0)
                else:
                    feature_vec.append(1)
                feature_vec.append(1)
                
                # custom features
                if L1 in end_punctuations:
                    feature_vec.append(True)
                else:
                    feature_vec.append(False)  
            
                # check if L is a salutation then it is not an EOS
                if R1 in start_punctuations:
                    feature_vec.append(True)
                else:
                    feature_vec.append(False) 
                        
                if len(L1.split('.')) > 1:
                    feature_vec.append(True)
                else:
                    feature_vec.append(False)
                X[c] = np.array(feature_vec)
                if psplit[2] == 'EOS':
                    Y.append(1)
                else:
                    Y.append(0)
                c += 1
            else:
                if 'EOS' in ele:
                    psplit = ele.split()
                    L = psplit[1]
                    psplit_next = data[i + 1].split()
                    if len(psplit_next) > 2:
                        R = psplit_next[1]
                    else:
                        R = ''
                    L = L.strip().lower()
                    R = R.strip().lower()
                    L1 = L; R1 = R
                    if L1 in salutation:
                        L1 = L1 + '.'
                    if R in salutation:
                        R1 = R1 + '.'
                    if L1 in self.dictionary.keys():
                        L_enc = self.dictionary[L1]
                    else:
                        L_enc = np.zeros((len(self.dictionary.values())))
                    if R1 in self.dictionary.keys():
                        R_enc = self.dictionary[R1]
                    else:
                        R_enc = np.zeros((len(self.dictionary.values())))
                    feature_vec = list(L_enc) + list(R_enc)
                    if len(L1) <= 3:
                        feature_vec.append(0)
                    else:
                        feature_vec.append(1)
                    
                    if L1 not in char_list and (ord(L1[0])>=97 and ord(L1[0])<=122):
                        feature_vec.append(0)
                    else:
                        feature_vec.append(1)
                
                        
                    if R1 not in char_list and (ord(R1[0])>=97 and ord(R1[0])<=122):
                        feature_vec.append(0)
                    else:
                        feature_vec.append(1)
                
                    # custom features
                    if L1 in end_punctuations:
                        feature_vec.append(True)
                    else:
                        feature_vec.append(False)  
            
                    # check if L is a salutation then it is not an EOS
                    if R1 in start_punctuations:
                        feature_vec.append(True)
                    else:
                        feature_vec.append(False) 
                        
                    if len(L1.split('.')) > 1:
                        feature_vec.append(True)
                    else:
                        feature_vec.append(False)
                     
                        
                    X[c] = np.array(feature_vec)
                    if psplit[2] == 'EOS':
                        Y.append(1)
                    else:
                        Y.append(0)
                    c += 1 
               
        return X, np.array(Y)

    def train(self, x_train, y_train):
        # clf is the classifier
        print("Training")
        self.clf.fit(x_train, y_train)
        return self.clf
    
    def accuracy(self, y_pred, y_true):
        acc_score = accuracy_score(y_pred, y_true)
        acc = 100 * acc_score
        return acc


if __name__ == "__main__":
    start = time.time()
    file1 = str(sys.argv[1])
    file2 = str(sys.argv[2])
    
    base = ''
    
    train_file = open(base + file1)
    test_file = open(base + file2)
    
    tr = train_file.read().split('\n')
    ts = test_file.read().split('\n')
    
           
    punctuations = ['?', '!', ',', '%', '"', '{', '}', '(', ')', '$', '#']
    numbers = ['1','2','3','4','5','6','7','8','9','0']
                    
    # create encoding
    print ("Create Encoding")
    words = []
    
    for ele in tr:
        if 'EOS' in ele:
            l = ele.split()
            if len(l) >= 2:
                if str(l[1]) not in punctuations:
                    words.append(str(l[1]).lower())
    
    for ele in ts:
        if 'EOS' in ele:
            l = ele.split()
            if len(l) >= 2:
                if str(l[1]) not in punctuations:
                    words.append(str(l[1]).lower())
            
    # unique words
    words = list(set(words))
    
    enc = preprocessing.LabelEncoder()
    new_words = enc.fit_transform(words)
    new_words = new_words.reshape(-1, 1)
    ohe = preprocessing.OneHotEncoder(sparse=False)
    encoded_words = ohe.fit_transform(new_words)
    dictionary = dict(zip(words, list(encoded_words)))
    
    # classifier intialization
    clf = DecisionTreeClassifier(random_state=0)
    s = SBD(clf, tr, ts, dictionary)
    
    print ("Prepping")
    x_tr, l_tr = s.prep('train')
    x_ts, l_ts = s.prep('test')
    s.clf = s.train(x_tr, l_tr)
    print('predicting')
    y_pred = s.clf.predict(x_ts)
    acc_my = s.accuracy(y_pred, l_ts)
    
    print ("The accuracy score for the prediction is")
    print (acc_my)
    end = time.time()
    
    print ("The total time taken is %d:" %(end-start))

    # Writing results onto a file
    c = 0
    print('Writing results onto a file')
    with open('SBD.test.out.txt', 'w+') as f:
        for i, ele in enumerate(ts):
            ele1 = ele.split()
            if 'TOK' in ele:
                f.write(ele1[0] + ' ' + ele1[1] + ' ' + 'TOK' + '\n')
            if 'EOS' in ele:
                a = ts[i].split()[1]
                if i == len(ts) - 1:
                    phrase = ts[i].split()[1]
                elif i < len(ts) - 1: 
                    a = ts[i].split()[1]
                    if len(ts[i + 1].split()) >= 2:
                        b = ts[i + 1].split()[1]
                        phrase = str(a) + ' ' + str(b)
                if y_pred[c] == 1:
                    f.write(ele1[0] + ' ' + ele1[1] + ' ' + 'EOS' + '\n')
                else:
                    f.write(ele1[0] + ' ' + ele1[1] + ' ' + 'NEOS' + '\n')
                c += 1

