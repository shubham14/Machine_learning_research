# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 19:40:27 2018

@author: Shubham
"""

import numpy as np
import scipy 
import sklearn
import random
import numpy.linalg as LA
from collections import defaultdict, Counter
import sys
from sklearn.metrics import mean_squared_error
import scipy.io as sio
import random
import pandas as pd
from scipy import sparse
from sklearn import preprocessing
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


np.random.seed(0)
# define Network class
# alpha is the learning rate
class Network:
    
    def __init__(self, inp, test_set, labels, dev_set, dev_labels, hidden_dim, alpha, epochs):
        self.inp = inp
        self.test_set = test_set
        self.labels = labels
        self.dev_set = dev_set
        self.dev_labels = dev_labels
        self.hidden_dim = hidden_dim
        self.inp_dim = inp.shape[1]
        self.out_dim = labels.shape
        self.W1 =  np.random.randn(hidden_dim, inp.shape[1]) * np.sqrt(2/inp.shape[1])
        self.hidden = np.random.randn(hidden_dim, 1)
        self.W2 = np.random.randn(labels.shape[1], hidden_dim) * np.sqrt(2/hidden_dim) 
#        self.b1 = np.zeros(self.hidden_dim, )
#        self.b2 = np.zeros(self.out_dim[1], )
        self.b1 = np.random.randn(self.hidden_dim, ) 
        self.b2 = np.random.randn(self.out_dim[1], )
        self.alpha = alpha
        self.epochs = epochs
        self.out = []
    
    # function definitions
    def sigmoid(self, z):
        if z.any() < -709:
            return 0
        return 1 / (1 + np.exp(-z))
    
    def sigmoidGrad(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def softmax(self, x):
        return np.exp(x - max(x)) / sum(np.exp(x - max(x))) 
        
    def softmaxGrad(self, x, targets):
        ans = []
        for j in range(3):
            sum1 = 0
            for i in range(3):
                if i == j:
                    sum1 += (x[i] - targets[i]) * x[i] * (1 - x[j])
                else:
                    sum1 += (x[i] - targets[i]) * x[i] * (-x[j])
            ans.append(sum1)
        return np.array(ans)        
            
    def MSE(self, x1, x2):
        return 0.5 * mean_squared_error(x1, x2)
    
    # for a single input
    def forward(self, inp1):
        l1 = self.sigmoid(np.dot(self.W1, inp1.T).T + self.b1)
#        out = self.softmax(np.dot(self.hidden, self.W2) + self.b2)
        out = self.softmax(np.dot(self.W2, l1.T).T + self.b2)
        ans_class = np.argmax(out)
        if ans_class == 0:
            x = np.array([1, 0, 0])
        elif ans_class == 1:
            x =  np.array([0, 1, 0])
        else:
            x = np.array([0, 0, 1])
        
        return out, x

    # for a single input
    def optimizeSGD(self, inp1, targets):
        l1 = self.sigmoid(np.dot(self.W1, inp1) + self.b1)
#        out = self.softmax(np.dot(self.hidden, self.W2) + self.b2)
        out = self.softmax(np.dot(self.W2, l1) + self.b2)
        l1 = np.expand_dims(l1, axis=1)

        # backpropogation 
        error_2 = (out - targets)
        del_soft = self.softmaxGrad(out, targets)
        layer2_W_delta = np.matmul(np.expand_dims(del_soft, axis = 1), l1.T)
        layer2_b_delta = del_soft
        
        error_1 = np.matmul(self.W2.T, del_soft) * self.sigmoidGrad(np.dot(self.W1, inp1) + self.b1)
        error_1_1 = np.expand_dims(error_1, axis=1)
        inp1 = np.expand_dims(inp1, axis=1)
        layer1_W_delta = np.matmul(error_1_1, inp1.T)
        layer1_b_delta = error_1
                
        self.W2 -= self.alpha * layer2_W_delta
        self.W1 -= self.alpha * layer1_W_delta 
        self.b2 -= self.alpha * layer2_b_delta
        self.b1 -= self.alpha * layer1_b_delta
        return out

        
    # train the network over number of epochs 
    # error list contains the loss function
    # batch size is kept at one
    def trainNetwork(self, target_file, train_text, dev_text, train_labels, dev_labels):
        acc_epoch_train = []; acc_epoch_dev = []; 
        for i in range(self.epochs):
            error = [];
            pred_class = []
            print ("Training at Epoch %d" %(i+1))
            for j in range(500000):
                print (j)
                batch = np.random.choice(self.inp.shape[0])
                X, y = self.inp[batch], self.labels[batch]
                out = self.optimizeSGD(X, y)
                train_loss = self.MSE(out, y)
#                if j % 1== 0:
#                	print (j, i)
#                	print('train loss: %0.6f' %train_loss)
                
                error.append(train_loss)
            
            pred_class_train = list(map(lambda x: self.predict_single(x).upper(), train_text))
            pred_class_dev = list(map(lambda x:self.predict_single(x).upper(), dev_text))
            
            acc_train = self.accuracy(train_labels, pred_class_train)
            acc_dev = self.accuracy(dev_labels, pred_class_dev)
            
            
            print ("At Epoch %d, the training accuracy is is %f" %(i+1, acc_train))
            acc_epoch_train.append(acc_train)
            
            print ("At Epoch %d, the dev accuracy is is %f" %(i+1, acc_dev))
            acc_epoch_dev.append(acc_dev)
   
        return acc_epoch_train, acc_epoch_dev, error
            
    # plotting loss
    def plot_loss(self, error, window_size):
        fig = plt.figure(figsize=(10, 10))
        s = np.convolve(error, np.ones((window_size,))/window_size, mode='valid')
        plt.plot(s)
        plt.show()
        fig.savefig('loss.png')
    
    # plotting accuracy
    def accuracy_plot(self, acc_train, acc_dev):
        fig = plt.figure(figsize=(10, 10))
        plt.plot(acc_train, label='training accuracy')
        plt.plot(acc_dev, label='dev accuracy')
        plt.legend()
        plt.show()
        fig.savefig('accuracy.png')
        
    # inp_enc_phrase is the collection of all 5-letter sequences in a given sentence
    def predict_single(self, inp_enc_phrase):
        a = 0
        ans = []
        for ele in inp_enc_phrase:
            step_1 = self.sigmoid(np.dot(self.W1, ele) + self.b1)
            step_2 = self.sigmoid(np.dot(self.W2, step_1) + self.b2)
            step_3 = self.softmax(step_2)
            ans_class = np.argmax(step_3)
            ans.append(ans_class)
        
        if len(ans) != 0:
            a = max(set(ans), key=ans.count)
        else:
            return str(-1)
        
        if a == 0:
            return 'English'
        elif a == 1:
            return 'French'
        else:
            return 'Italian'
    
    # train_text and dev_set are parsed from process_inp_tr
    def predict(self, mode, train_text, dev_text):
        if mode == 'test':
            print('Test predicting')
            pred_lang = list(map(lambda x: self.predict_single(x), self.test_set))
            return pred_lang
        elif mode == 'train':
            print('Train predicting')
            pred_lang = list(map(lambda x: self.predict_single(x).upper(), train_text))
            return pred_lang
        elif mode == 'dev':
            print('dev predicting')
            pred_lang = list(map(lambda x: self.predict_single(x).upper(), dev_text))
            return pred_lang
        
    # can be used for all test, train and development files
    def accuracy(self, target_file, pred_lang):
        print('Calculating accuracy')
#        target_file = open('test_solutions').read().split('\n')
        c = 0
        for i in range(len(pred_lang[:-1])):
            if pred_lang[i].lower() == target_file[i].lower():
                c += 1
        ans = c / len(target_file)
        return ans
            
    
# data prepping for neural network  
# char list is a unique set of characters used in any language
class DataLoader:
    def __init__(self):
        self.label_dict = {'ENGLISH':[1,0,0], 'FRENCH':[0,1,0], 'ITALIAN':[0,0,1]}
        self.char_dict = defaultdict(int)
        self.char_list = []
        self.len_data = 7781
    
    def create_dict(self):
        tr_file = open('train', encoding='ISO-8859-1')
        ts_file = open('test', encoding='ISO-8859-1')
        dev_file = open('dev', encoding='ISO-8859-1')
        tr_data_split = tr_file.read().split('\n')
        ts_data_split = ts_file.read().split('\n')
        dev_data_split = dev_file.read().split('\n')
        
        data_tr = list(' '.join(tr_data_split))
        c_letters = list(map(lambda x: x.lower(), data_tr))
        
        data_ts = list(' '.join(ts_data_split))
        c_letters_1 = list(map(lambda x: x.lower(), data_ts))
        
        data_dev = list(' '.join(dev_data_split))
        c_letters_2 = list(map(lambda x: x.lower(), data_dev))
        
        c_l = c_letters + c_letters_1 + c_letters_2 
        self.char_list = list(set(c_l))
        
    
    # change parse data for incorporating test data
    def parse_data(self, filename, label_exp):
        labels = []
        text = []
        data_new = open(filename, encoding='ISO-8859-1')
        data_split = data_new.read().split('\n')
        if filename == 'train' or filename == 'dev':
            if label_exp:
                for ele in data_split:
                    if len(ele) != 0:
                        ele1 = ele.split()
                        labels.append(self.label_dict[ele1[0]])
                        text.append(' '.join(ele1[1:]))
            else:
                for ele in data_split:
                    if len(ele) != 0:
                        ele1 = ele.split()
                        labels.append(ele1[0])
                        text.append(' '.join(ele1[1:]))
            return np.array(text), np.array(labels)
        elif filename == 'test':
            return np.array(data_split)
        
    def encode_data(self):
        enc = preprocessing.LabelEncoder()
        new_letters = enc.fit_transform(self.char_list)
        new_letters = new_letters.reshape(-1, 1)
        ohe = preprocessing.OneHotEncoder(sparse=False)
        encoded_letters = ohe.fit_transform(new_letters)
        self.char_dict = dict(zip(self.char_list, list(encoded_letters)))
    
    # processing input to create a 5-c dimensional vector for the neural network
    # for a single phrase
    # this is for the training process
    def process_inp_tr(self, single_phrase, label):
         punctuations = [',', '_', '-']
         s2 = list(single_phrase)
#         random.shuffle(s2)
         l = []
         d = list(filter(lambda x: x not in punctuations, s2))
         for i in range(len(d) - 4):
             l1 = list(map(lambda x: self.char_dict[x.lower()], d[i : i + 5]))
             flattened_l1 = np.array([val for sublist in l1 for val in sublist])
             l.append(flattened_l1)
         return l, [label] * len(l)
     
    # for testing process
    def process_inp_ts(self, single_phrase):
         punctuations = [',', '_', '-']
         s2 = list(single_phrase)
         l = []
         d = list(filter(lambda x: x not in punctuations, s2))
         for i in range(len(d) - 4):
             l1 = list(map(lambda x: self.char_dict[x.lower()], d[i : i + 5]))
             flattened_l1 = np.array([val for sublist in l1 for val in sublist])
             l.append(flattened_l1)
         return l
     
    def processed_inp(self, mode, ordered):
        if mode == 'train' or mode == 'dev':
            t, l = self.parse_data(mode, True)
            processed_text = []
            processed_labels = []
            self.encode_data()
            for i in range(len(t)):
                if i % 100 == 0:
                    print (i)
                t1, l1 = self.process_inp_tr(t[i], l[i])
                processed_text.append(t1)
                processed_labels.append(l1)
            if ordered == False:
                flattened_text = np.array([val for sublist in processed_text for val in sublist], dtype=np.uint8)
                flattened_labels = np.array([val for sublist in processed_labels for val in sublist], dtype=np.uint8)
                return flattened_text, flattened_labels
            elif ordered == True:
                return processed_text, processed_labels
        
        elif mode == 'test':
            p_list = []
            t = self.parse_data(mode, True)
            for j in range(len(t)):
                print (j)
                t1 = self.process_inp_ts(t[j])
                p_list.append(t1)
            return p_list

    
if __name__ == "__main__":
    # data preparation for neural network
    data = DataLoader()
    data.create_dict()
    data.encode_data()
    
    # processed and encoded train and test data
    print('Parsing Train data')
    text_tr, labels_tr = data.processed_inp('train', False)
    print('Parsing test data')
    text_ts = data.processed_inp('test', False)
    
    # for accuracy measures
    text_tr_ordered, _ = data.processed_inp('train', True)
    print('Parsing Dev data')
    text_dev, _ = data.processed_inp('dev', True)
        
    _, labels_tr_ordered = data.parse_data('train', False)
    _, labels_dev = data.parse_data('dev', False)
#    
#     network initialization
    alpha = 0.1
    hidden_dim = 100
    st = time.time()
    net = Network(text_tr, text_ts, labels_tr, text_dev, labels_dev, hidden_dim, alpha, 3)
    
    # train network
    target_file = open('test_solutions', encoding='ISO-8859-1').read().split('\n')
    targets_test = []
    for ele in target_file:
        ele1 = ele.split()
        if len(ele1) > 1:
            targets_test.append(ele1[1])
    acc_epoch_train, acc_epoch_dev, error = net.trainNetwork(target_file, text_tr_ordered,
                                                             text_dev, labels_tr_ordered,
                                                             labels_dev)
    
    # prediction and accuracy calculation
    pred = net.predict('test', text_tr_ordered, text_dev)
    
#    target_file = open('test_solutions').read().split('\n')
    test_acc = net.accuracy(pred, targets_test)
    print (test_acc)
    end = time.time()
    
    # writing results onto a file
    p = open('test', encoding='ISO-8859-1').read().split('\n')
    with open('languageIdentificationPart1.output.txt', 'w+') as w:
        for i in range(300):
            w.write('Line%d %s ' %(i+1, pred[i]))
            w.write('\n')
     
    total_time = end - st
    print("The total time taken is : %d" %total_time)