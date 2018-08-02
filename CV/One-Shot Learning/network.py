# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:00:57 2018

@author: Shubham
"""

# import keras utility libraries
import keras 
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, Activation
from keras.layers import Lambda, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate, _Merge
from keras.utils import plot_model
from keras.optimizers import *
import keras.backend as K

# custom layer for calculating the absolute difference 
# between two tensors
class Sub(_Merge):
    def build(self, input_shape):
        super(Sub, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A Subtract Layer is called on exactly 2 inputs')
            
    def _merge_function(self, inputs):
        if len(input_shape) != 2:
            raise ValueError('A Subtract Layer is called on exactly 2 inputs')
        return K.abs(inputs[0] - inputs[1])

# class containing the Siamese Network Architecture
class SiameseNetWork:
    
    # define parameters
    def __init__(self, n_filters, n_rows, n_cols, n_ch, fc_units,
                 beta1, beta2, lr=1e-3, n_out_ch=1, **kwargs):
        self.n_filters = n_filters
        self.lr = lr
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_ch = n_ch
        self.fc_units = fc_units
        self.beta1 = beta1
        self.beta2 = beta2
    
    def net1(self):
        inp = Input(shape=(self.n_rows, self.n_cols, self.n_ch))
        x = Conv2D(self.n_filters[0], (10, 10), strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(self.n_filters[1], (7, 7), strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(self.filters[2], (4, 4), strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(self.filters[3], (4, 4), strides=1, padding='same')(x)
        x = Flatten()(x)
        g_out = Dense(self.fc_units, activation='sigmoid')(x)
        Net1 = Model(inp, g_out, name='network1')
        return Net1
    
    def net2(self):
        inp = Input(shape=(self.n_rows, self.n_cols, self.n_ch))
        x = Conv2D(self.n_filters[0], (10, 10), strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(self.n_filters[1], (7, 7), strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(self.filters[2], (4, 4), strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(self.filters[3], (4, 4), strides=1, padding='same')(x)
        x = Flatten()(x)
        g_out = Dense(self.fc_units, activation='sigmoid')(x)
        Net2 = Model(inp, g_out, name='network2')
        return Net2
        
    def diff_net(self):
        # instantiating the networks
        inp1 = Input(shape=self.net1.layers[0].input_shape[1:])
        inp2 = Input(shape=self.net2.layers[0].inout_shape[1:])
        net1_output = self.net1(inp1)
        net2_output = self.net2(inp2)
        sub = Sub()([net1_output, net2_output])
        sub_out = Activation('sigmoid')(sub)
        siamese_net = Model([inp1, inp2], sub_out)
        opt = Adam(lr=self.lr, beta1=self.beta1, beta2=self.beta2)
        siamese_net.compile(optimizer=opt, loss='binary_crossentropy')
        return siamese_net