# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 20:04:26 2018
Contains Utility functions required for the main.py, plotting, noise among others
@author: Shubham
"""


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
from PIL import Image

def generate_images(generator_model, output_dir, epoch, n_class, batch_size=128):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG file."""
    label = np.random.randint(0, n_class, size=(batch_size, 1))
    label = np_utils.to_categorical(label, num_classes=n_class)
    test_image_stack = generator_model.predict([np.random.rand(10, 100), label])
    test_image_stack = (test_image_stack * 127.5) + 127.5
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output, mode='L')  # L specifies greyscale
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)
    
    
def plot_loss(losses, save_file=False, file_name=None):
    plt.figure(figsize=(10, 10))
    g_loss = np.array(losses['g'])
    d_loss = np.array(losses['d'])
    plt.plot(g_loss[:, 1], label='G loss')
    plt.plot(g_loss[:, 2], label='Q loss')
    plt.plot(d_loss[:, 1], label='D loss')
    plt.plot(d_loss[:, 2], label='D mse')
    plt.legend()
    if save_file:
        if file_name is not None:
            plt.savefig(file_name)
            plt.show()
    plt.clf()
    plt.close('all')


def get_noise(dim_noise, batch_size=128):
    noise = np.random.uniform(-1, 1, size=(batch_size, dim_noise))
    label = np.random.randint(0, n_class, size=(batch_size, 1))
    label = np_utils.to_categorical(label, num_classes=n_class)
    return noise, label


# plot generated images
def plot_gen(generator, dim=(4, 4), figsize=(10, 10), channel=0, save=False, saveFileName=None, **kwargs):
    dim_noise = generator.layers[0].input_shape[1]
    n_image_col = dim[1]
    plt.figure(figsize=figsize)
    for i in range(n_class):
        noise, _ = get_noise(dim_noise, batch_size=n_image_col)
        label = np.repeat(i, n_image_col).reshape(-1, 1)
        label = np_utils.to_categorical(label, num_classes=n_class)
        image_gen = generator.predict([noise, label])
        image_gen = (image_gen * 127.5) + 127.5
        for j in range(n_image_col):
            plt.subplot(n_class, n_image_col, i * n_image_col + j + 1)
            plt.imshow(image_gen[j, :, :, channel], cmap="gray", **kwargs)
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
def set_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val