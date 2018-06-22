# -*- coding: utf-8 -*-
"""
Created on Sun May  6 11:10:43 2018

@author: Shubham
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import sys

# read the MNIST data
mnist = input_data.read_data_sets('MNIST_data')

# Network parameters
batch_size = 64
learning_rate = 0.0005
nepoch = 30000

tf.reset_default_graph()

X_in = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name = 'X')
Y = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name = 'Y')
Y_flat = tf.reshape(Y, shape = [-1, 28*28])
keep_prob = tf.placeholder(dtype = tf.float32, shape = (), name = 'keep_prob')

dec_in_channels = 1
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = int(49 * dec_in_channels/2)

class VAE:
    
    def __init__(self, n_latent):
        self.n_latent = n_latent
    
    # alpha is a variable dictating the "leakiness" of the relu
    def lrelu(self, x, alpha = 0.3):
        return tf.maximum(x, tf.multiply(x, alpha))
    
    # Note that the architecture can be changed accordingly
    # Encoder function
    def encoder(self, X_in, keep_prob):
        activation = self.lrelu
        with tf.variable_scope("encoder", reuse = False):
            x = tf.reshape(X_in, shape = [-1, 28, 28, 1])
            x = tf.layers.conv2d(x, filters = 64, kernel_size = 4, strides = 2, padding = 'same', activation = activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, filters = 64, kernel_size = 4, strides = 2, padding = 'same', activation = activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.contrib.layers.flatten(x)
            mn = tf.layers.dense(x, units = n_latent)
            sd = 0.5 * tf.layers.dense(x, units = n_latent)
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
            z = mn + tf.multiply(epsilon, tf.exp(sd))
            return z, mn, sd
        
    # Decoder function
    def decoder(self, sampled_z, keep_prob):
        activation = self.lrelu
        with tf.variable_scope("decoder", reuse = False):
            x = tf.layers.dense(sampled_z, units = inputs_decoder, activation = tf.nn.relu)
            x = tf.layers.dense(x, units = inputs_decoder * 2 + 1, activation = tf.nn.relu)
            x = tf.reshape(x, reshaped_dim)
            x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 4, strides = 2,
                                           padding = 'same', activation = tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 4, strides = 1, 
                                           padding = 'same', activation = tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 4, strides = 1,
                                           padding = 'same', activation = tf.nn.relu)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units = 28*28, activation = tf.nn.sigmoid)
            img = tf.reshape(x, shape = [-1, 28, 28])
            return img
        

if __name__ == "__main__":
    vae = VAE(n_latent)
    sampled, mean, std_dev = vae.encoder(X_in, keep_prob)
    dec = vae.decoder(sampled, keep_prob)
    
    unreshaped = tf.reshape(dec, [-1, 28*28])
    img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
    latent_loss = -0.5 * tf.reduce_sum(1. + 2. * std_dev - tf.square(mean) - tf.exp(2.0 * std_dev), 1)
    loss = tf.reduce_mean(img_loss + latent_loss)
    
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        for i in range(nepoch):
            data_batch = [np.reshape(batch, [28, 28]) for batch in mnist.train.next_batch(batch_size = batch_size)[0]]
            sess.run(init)
            sess.run(optimizer, feed_dict = {X_in: data_batch, Y: data_batch, keep_prob: 0.8})
            
            if i % 500 == 0:
                ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mean, std_dev], feed_dict = {X_in: data_batch,
                                                       Y: data_batch, keep_prob: 1.0})
                plt.imshow(np.reshape(data_batch[0], [28, 28]), cmap='gray')
                plt.show()
                plt.imshow(d[0], cmap='gray')
                plt.show()
                print(i, ls, np.mean(i_ls), np.mean(d_ls))
            
            randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
            imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
            imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]
        
            for img in imgs:
                plt.figure(figsize=(1,1))
                plt.axis('off')
                plt.imshow(img, cmap='gray')
