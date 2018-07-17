# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 23:06:47 2018

@author: Shubham
"""

# Contains all the utility functions
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from os import path
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import h5py
import random
from tqdm import tqdm
from IPython import display

K.set_image_data_format('channels_last')
K.set_image_dim_ordering('tf')

class Utils():
    
    def __init__(self, loss_list, save_file=False, file_name=None, dim_noise, batch_size=128, n_class):
        self.save_file = save_file
        self.file_name = file_name
        self.loss_list = loss_list
        self.dim_noise = dim_noise
        self.batch_size = batch_size
        self.n_class = n_class
    
    def plot_loss(self):
        plt.figure(figsize=(10, 10))
        g_loss = np.array(self.loss_list['g'])
        d_loss = np.array(self.loss_list['d'])
        plt.plot(g_loss[:, 1], label='G loss')
        plt.plot(g_loss[:, 2], label='Q loss')
        plt.plot(g_loss[:, 1], label='D loss')
        plt.plot(g_loss[:, 2], label='D mse')
        plt.legend()
        if save_file:
            if file_name is not None:
                plt.savefig(self.file_name)
                plt.show()
        plt.clf()
        plt.close('all')
    
    def get_noise(self):
        noise = np.random.uniform(-1, 1, size=(self.batch_size, self.dim_noise))
        label = np.random.randint(0, self.n_class, size=(self.batch_size, 1))
        label = np_utils.to_categorical(label, num_classes=self.n_class)
        return noise, label
    
    # plot generated images
    def plot_gen(self, generator, dim=(4, 4), figsize=(10, 10), channel=0, save=False, saveFileName=None, **kwargs):
        dim_noise = generator.layers[0].input_shape[1]
        n_image_col = dim[1]
        plt.figure(figsize=figsize)
        for i in range(n_class):
            noise, _ = get_noise(dim_noise, batch_size=n_image_col)
            label = np.repeat(i, n_image_col).reshape(-1, 1)
            label = np_utils.to_categorical(label, num_classes=n_class)
            image_gen = generator.predict([noise, label])
            for j in range(n_image_col):
                plt.subplot(n_class, n_image_col, i * n_image_col + j + 1)
                plt.imshow(image_gen[j, :, :, channel], **kwargs)
                plt.axis('off')
        if save:
            if saveFileName is not None:
                plt.savefig(saveFileName)
        else:
            plt.tight_layout()
            plt.show()
        plt.clf()
        plt.close('all')
    
    # freeze weights in the discriminator for stacked training
    def set_trainable(self, net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val