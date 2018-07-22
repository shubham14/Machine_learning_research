# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 20:20:06 2018
Contains customized loss functions for  
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

# import keras packages
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, Activation, Lambda, GlobalAveragePooling2D
from keras.layers.convolutional import Conv1D, Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import LeakyReLU, ReLU, ELU
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.layers.merge import _Merge
from keras.utils import plot_model
from keras.optimizers import *
from keras.engine.topology import Layer
from keras import backend as K
from functools import partial
from keras.losses import *
from keras.layers.core import *
import tensorflow as tf
from PIL import Image

K.set_image_data_format('channels_last')
K.set_image_dim_ordering('tf')

def wasser_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    grad = K.gradients(y_pred, averaged_samples)[0]
    grad_sqr = K.square(grad)
    grad_sqr_sum = K.sum(grad_sqr, axis=np.arange(1, len(grad_sqr.shape)))
    grad_l2_norm = K.sqrt(K.abs(grad_sqr_sum))
    grad_penalty = gradient_penalty_weight * K.square(1 - grad_l2_norm)
    return K.mean(grad_penalty)


# Loss layers defined as classes
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
