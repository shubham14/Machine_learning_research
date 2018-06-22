# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:37:18 2018

@author: Shubham
"""

import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K

class Net():
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
    
    # Sequential Model
    def make_generator(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.input_dim))
        model.add(LeakyReLU())
        model.add(Dense(128 * 7 * 7))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        if K.image_data_format() == 'channels_first':
            model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
            bn_axis = 1
        model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
        model.add(BatchNormalization(axis=bn_axis))
        model.add(LeakyReLU())
        model.add(Convolution2D(64, (5, 5), padding='same'))
        model.add(BatchNormalization(axis=bn_axis))
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
        model.add(BatchNormalization(axis=bn_axis))
        model.add(LeakyReLU())
        model.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))
        return model


    def make_discriminator(self):
        model = Sequential()
        if K.image_data_format() == 'channels_first':
            model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(1, 28, 28)))
        model.add(LeakyReLU())
        model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2]))
        model.add(LeakyReLU())
        model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(1024, kernel_initializer='he_normal'))
        model.add(LeakyReLU())
        model.add(Dense(1, kernel_initializer='he_normal'))
        return model