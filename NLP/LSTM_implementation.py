# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 01:10:17 2018

@author: Shubham
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import random
import time
from collections import Counter

# log path
#logs_path = '/tmp/tensorflow/rnn_words'
#writer = tf.summary.FileWriter(logs_path)

# training file
training_file = 'belling_the_cat.txt'
logs_path = 'C\\Users\\Shubham\\Desktop'
writer = tf.summary.FileWriter(logs_path)

class DataPrepper:
    def __init__(self, fname):
        self.fname = fname
        self.dict = dict()
        self.reverse_dict = dict()
        self.vocab_size = 0
        
    def read_data(self):
        with open(self.fname) as f:
            content = f.readlines()
        content = content.split()
        return np.array(content)
    
    def build_dataset(self, words):
        count = Counter(words).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
            reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        self.dict = dictionary
        self.reverse_dict = reverse_dictionary
        self.vocab_size = len(self.dict)
        
    def encodeData(self):
        words = self.read_data()
        self.build_dataset(words)
        
    
class Network(DataPrepper):
    def __init__(self, lr, tr_iters, display_step, n_input,
                 n_hidden):
        self.lr = lr
        self.tr_iters = tr_iters
        self.display_step = display_step
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.x = tf.placeholder("float", [None, n_input, 1])
        self.y = tf.placeholder("float", [None, vocab_size])
        self.weights = {'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))}
        self.biases = {'out': tf.Variable(tf.random_normal([self.vocab_size]))}
        
    def RNN(self):
        self.x = tf.reshape(x, [-1, n_input])
        x = tf.split(x, n_input, 1)
        rnn_cell = rnn.BasicLSTMCell(n_hidden)
        output, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
        
        # since we require the last outputs
        return tf.matmul(output[-1], weights['out']) + biases['out']
    
    def train(self, data, dictionary):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(cost)
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            step = 0
            offset = random.randint(0, self.n_input + 1)
            end_offset = self.n_input + 1
            acc_total = 0
            loss_total = 0
            
            writer.add_graph(sess.graph)
            
            while step < self.tr_iters:
                if offset > (len(data) - end_offset):
                    offset = random.randint(0, self.n_input + 1)
                
                symbols_in_keys = [[dictionary[str(data[i])]] for i in range(offset, offset + self.n_input)]
                symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, self.n_input, 1])
                
                # one hot encoding
                symbols_out_onehot = np.zeros([-1, self,n_input, 1])
                symbols_out_onehot[dictionary[str(data[offset + self.n_input])]] = 1.0
                symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])
                
                _, acc, loss, onehot_pred = sess.run([optimizer, accuracy, cost, pred],
                                                     feed_dict:{x: symbols_in_keys, y:symbols_out_one})
                
                loss_total += loss
                acc_total += acc
                if (step+1) % display_step == 0:
                    print("Iter= " + str(step+1) + ", Average Loss= " + \
                          "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                          "{:.2f}%".format(100*acc_total/display_step))
                acc_total = 0
                loss_total = 0
                symbols_in = [data[i] for i in range(offset, offset + self.n_input)]
                symbols_out = training_data[offset + self.n_input]
                symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                step += 1
                offset += (self.n_input + 1)
                
                
if __name__ == "__main__":
    data = DataPrepper('train')