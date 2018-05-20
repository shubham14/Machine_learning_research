# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:46:47 2018

@author: Shubham
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import rnn_cell
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from tensorflow.examples.tutorials.mnist import input_data

# mnist load data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# training parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# network parameter
num_input = 28
timesteps = 28
num_hidden = 128

# 10 classes for mnist classification
num_classes = 10

# 1 for single RNN cell and 2 for stacked RNN cell
choice = int(input())

#tf Graph inputX
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
        'out': tf.Variable(tf.random.normal([num_hidden, num_classes]))
        }

biases = {
        'out': tf.Variable(tf.random.normal([num_classes]))
        }

def RNN(X, weights, biases):
    
    x = tf.unstack(x, timesteps, 1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    output, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']    
    

logits = RNN(X, weights, biases)
pred = tf.nn.softmax(logits)

# optimizer variables
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimzer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# model evaluation parameters
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step%display_step == 0 or step == 1:
            
            # printing the values of loss and accuracy at a particular time interval
            # Oh!! Keras is far better in this aspect
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))