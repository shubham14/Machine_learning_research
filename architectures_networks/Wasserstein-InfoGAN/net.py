# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 22:30:00 2018

@author: Shubham
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from os import path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import h5py
import random
from tqdm import tqdm
from IPython import display

# import keras packages
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, Activation, Lambda, GlobalAveragePooling2D
from keras.layers.convolutional import Conv1D, Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.utils import plot_model  
from keras.optimizers import *
from keras import backend as K
K.set_image_dim_ordering('tf')
from losses import *

class Net():
    
    def __init__(self, n_rows, n_cols, n_out_ch=1, n_first_conv_ch=128, dim_noise=100, dim_cat=n_class,
                  n_in_ch=1, n_last_conv_ch=128, leaky_relu_alpha=0.2, lr=1e-4, beta_1=0.5, beta_2=0.9):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_out_ch = n_out_ch
        self.n_in_ch = n_in_ch
        self.n_first_conv_ch = n_first_conv_ch
        self.last_conv_ch = last_conv_ch
        self.dim_noise = dim_noise
        self.dim_cat = dim_cat
        self.leaky_relu_alpha = leaky_relu_alpha
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    # functional method of forming the model
    def build_generator(self):
        g_in_noise = Input(shape=(self.dim_noise,))
        g_in_cat = Input(shape=(self.dim_cat,))
        g_in = concatenate([g_in_noise, g_in_cat])
        x = Dense(self.n_first_conv_ch * int(self.n_rows) * int(self.n_cols))(g_in)
        x = Reshape((int(self.n_rows), int(self.n_cols), self.n_first_conv_ch))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(int(self.n_first_conv_ch / 2), (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(int(self.n_first_conv_ch / 4), (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(int(self.n_first_conv_ch / 8), (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        g_out = Conv2DTranspose(self.n_out_ch, (3, 3), strides=1, padding='same', activation='tanh')(x)
        generator = Model([g_in_noise, g_in_cat], g_out, name='generator')
        print ('Summary of Generator (for InfoGAN)')
        generator.summary()
        return generator
    
    
    def build_discriminator(self):
        d_opt = Adam(lr=self.lr, beta_1=self.beta_1)
        d_in = Input(shape=(self.n_rows, self.n_cols, self.n_in_ch))
        x = Conv2D(int(self.n_last_conv_ch / 8), (3, 3), strides=1, padding='same')(d_in)
        x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = Conv2D(int(self.n_last_conv_ch / 4), (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = Conv2D(int(self.n_last_conv_ch / 2), (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = Conv2D(self.n_last_conv_ch, (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha, name='d_feature')(x)
        d_out_base = Flatten()(x)
    
        # real probability output
        d_out_real = Dense(1)(d_out_base)
    
        # categorical output
        x = Dense(128)(d_out_base)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        d_out_cat = Dense(self.n_class, activation='sigmoid')(x)
    
        discriminator = Model(d_in, d_out_real, name='discriminator')
        discriminator.compile(optimizer=d_opt, loss='binary_crossentropy', metrics=['mean_squared_error'])
        print ('Summary of Discriminator (for InfoGAN)')
        discriminator.summary()
    
        auxiliary = Model(d_in, d_out_cat, name='auxiliary')
        auxiliary.compile(optimizer=d_opt, loss='categorical_crossentropy', metrics=['mean_squared_error'])
        print ('Summary of Auxiliary (for InfoGAN)')
        auxiliary.summary()
    
        return discriminator, auxiliary
    
    
    def infogan_model_wrapper_new(self, generator, discriminator, auxiliary, d_lr=2e-4):
        d_opt = Adam(lr=d_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        set_trainable(discriminator, False)
        real_samples = Input(shape=discriminator.layers[0].input_shape[1:])
        generator_input_for_discriminator = Input(shape=generator.layers[0].input_shape[1:])
        gen_cat_inp_disc = Input(shape=generator.layers[1].input_shape[1:])
        gen_out = generator([generator_input_for_discriminator, gen_cat_inp_disc])
    
        averaged_samples = loss.RandomWeightedAverage()([real_samples, gen_out])
        averaged_samples_out = discriminator(averaged_samples)
        sub = Subtract()([discriminator(gen_out), discriminator(real_samples)])
        norm = GradNorm()([averaged_samples_out, averaged_samples])

        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=LAMBDA)
        partial_gp_loss.__name__ = 'gradient penalty'
    
        discriminator_model = Model(inputs=[real_samples,
                                            generator_input_for_discriminator,
                                            gen_cat_inp_disc],
                                    outputs=[sub, norm])
        set_trainable(discriminator, True)
        discriminator_model.compile(optimizer=d_opt, loss=[wasser_loss, 'mse'],
                                    loss_weights=[1, 1])
        auxiliary.compile(optimizer=d_opt, loss='categorical_crossentropy')
        return generator, discriminator_model, auxiliary
    
    
    def build_infogan(self, generator, discriminator, discriminator_model,
                      auxiliary, g_lr=1e-4):
        gan_opt = Adam(lr=g_lr, beta_1=self.beta_1)
        real_samples = Input(shape=discriminator.layers[0].input_shape[1:])
        gan_in_noise = Input(shape=generator.layers[0].input_shape[1:])
        gan_in_cat = Input(shape=generator.layers[1].input_shape[1:])
        gen_out = generator([gan_in_noise, gan_in_cat])
    
        set_trainable(discriminator_model, False)
        set_trainable(generator, True)
        [sub, norm] = discriminator_model([real_samples, gan_in_noise,
                                                             gan_in_cat])
        gan_out_cat = auxiliary(gen_out)
        gan = Model([real_samples, gan_in_noise, gan_in_cat],
                    [sub, norm, gan_out_cat])
        gan.compile(optimizer=gan_opt, loss=['mse', 'mse', 'mse'],
                    loss_weights=[1, 1, 1])
        gan.summary()
        return gan
