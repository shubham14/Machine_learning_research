# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 20:08:16 2018

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
from losses import *
K.set_image_data_format('channels_last')
K.set_image_dim_ordering('tf')

# hyper parameters
n_class = 10
LAMBDA = 10
EPSILON = 1e-8
n_critic = 5
output_dir = '/home/SDash2/Desktop/MNIST_Output/'
n_filters = [128, 64, 32, 16]  # for a four layer CNN generator-critic


class Net:
    
    def __init__(self, n_rows, n_cols, n_out_ch=1, n_first_conv_ch=128, dim_noise=100, dim_cat=n_class, lr=1e-4,
                 beta_1=0.5, leaky_relu_alpha=0.2, n_in_ch=1):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_out_ch = n_out_ch
        self.n_first_conv_ch = n_first_conv_ch
        self.dim_noise = dim_noise
        self.dim_cat = dim_cat
        self.lr = lr
        self.beta_1 = beta_1
        self.leaky_relu_alpha = leaky_relu_alpha
        self.n_in_ch = n_in_ch
        
    def build_generator(self):
        g_in_noise = Input(shape=(self.dim_noise,))
        g_in_cat = Input(shape=(self.dim_cat,))
        g_in = concatenate([g_in_noise, g_in_cat])
        x = Dense(self.n_first_conv_ch * int(self.n_rows) * int(self.n_cols))(g_in)
        x = Reshape((int(self.n_rows), int(self.n_cols), self.n_first_conv_ch))(x)
        x = BatchNormalization(axis=-1, momentum=0.8)(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = Conv2DTranspose(int(self.n_first_conv_ch / 2), (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization(axis=-1, momentum=0.8)(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = Conv2DTranspose(int(self.n_first_conv_ch / 4), (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization(axis=-1, momentum=0.8)(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = Conv2DTranspose(int(self.n_first_conv_ch / 8), (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization(axis=-1, momentum=0.8)(x)
        g_out = Conv2DTranspose(self.n_out_ch, (3, 3), strides=1, padding='same', activation='tanh')(x)
        generator = Model([g_in_noise, g_in_cat], g_out, name='generator')
        print ('Summary of Generator (for InfoGAN)')
        generator.summary()
        return generator


    def build_discriminator(self):
        d_opt = Adam(lr=self.lr, beta_1=self.beta_1)
        d_in = Input(shape=(self.n_rows, self.n_cols, self.n_in_ch))
        x = Conv2D(int(self.n_last_conv_ch / 2), (3, 3), padding='same')(d_in)
        x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = Conv2D(int(self.n_last_conv_ch), (3, 3), strides=2)(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = Conv2D(self.n_last_conv_ch, (3, 3), strides=2, padding='same')(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = Flatten()(x)
        d_out_base = Dense(1024)(x)
        d_out_disc = LeakyReLU(alpha=self.leaky_relu_alpha)(d_out_base)
    
        # real probability output
        d_out_real = Dense(1)(d_out_disc)
    
        # categorical output
        x = Dense(1024)(d_out_base)
        x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        d_out_cat = Dense(self.n_class, activation='softmax')(x)
    
        discriminator = Model(d_in, d_out_real, name='discriminator')
        print ('Summary of Discriminator (for InfoGAN)')
        discriminator.summary()
    
        auxiliary = Model(d_in, d_out_cat, name='auxiliary')
        print ('Summary of Auxiliary (for InfoGAN)')
        auxiliary.summary()
    
        return discriminator, auxiliary
    

class InfoGAN:
    def __init__(generator, discriminator, auxiliary):
        self.generator = generator
        self.discriminator = discriminator
        self.auxiliary = auxiliary
    
    def build_infogan(self, g_lr=1e-4, beta_1=0.5, beta_2=0.9):
        d_opt = Adam(lr=d_lr, beta_1=beta_1, beta_2=beta_2)
        gan_opt = Adam(lr=g_lr, beta_1=beta_1, beta_2=beta_2)
        set_trainable(self.discriminator, False)
        real_samples = Input(shape=self.discriminator.layers[0].input_shape[1:])
        generator_input_for_discriminator = Input(shape=self.generator.layers[0].input_shape[1:])
        gen_cat_inp_disc = Input(shape=self.generator.layers[1].input_shape[1:])
        gen_out = self.generator([generator_input_for_discriminator, gen_cat_inp_disc])
        dis_gen = self.discriminator(gen_out)
    
        gan_out_cat = self.auxiliary(gen_out)
        gan = Model([real_samples, generator_input_for_discriminator, gen_cat_inp_disc],
                    [dis_gen, gan_out_cat])
        gan.compile(optimizer=gan_opt, loss=[wasser_loss,
                                             'categorical_crossentropy'],
                    loss_weights=[1, 1])
    
        set_trainable(self.discriminator, True)
        set_trainable(self.generator, False)
    
        averaged_samples = RandomWeightedAverage()([real_samples, gen_out])
        averaged_samples_out = self.discriminator(averaged_samples)
    
        dis_real = self.discriminator(real_samples)
        dis_gen = self.discriminator(gen_out)
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=LAMBDA)
        partial_gp_loss.__name__ = 'gradient penalty'
    
        discriminator_model = Model(inputs=[real_samples,
                                            generator_input_for_discriminator,
                                            gen_cat_inp_disc],
                                    outputs=[dis_real, dis_gen,
                                             averaged_samples_out])
        discriminator_model.compile(optimizer=d_opt, loss=[wasser_loss,
                                                           wasser_loss,
                                                           partial_gp_loss],
                                    loss_weights=[1, 1, 1])
    
        gan.summary()
        return discriminator_model, gan