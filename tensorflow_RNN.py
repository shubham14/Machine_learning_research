# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:46:47 2018

@author: Shubham
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# network parameters
num_epochs = 100
total_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_length//batch_size//truncated_backprop_length

def generateData():
    x = np.array(np.random.choice(2, total_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))
    
    return (x, y)

# input-output placeholder
batchX_palceholder = tf.placeholder(tf.float32, [batch_size, 
                                                 truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.float32, [batch_size,
                                                 truncated_backprop_length])
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

# Variables to be optimized
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Varaiable(np.zeros((1, state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Varaiable(np.zeros((1, num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unpack(batchX_placeholder, axis=1)
labels_series = tf.unpack(batchY_placeholder, axis=1)