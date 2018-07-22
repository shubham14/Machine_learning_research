# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:30:46 2018
main driver function for wasserstein infogan for 
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

n_class = 10
LAMBDA = 10
EPSILON = 1e-8
n_critic = 5
output_dir = '/home/SDash2/Desktop/MNIST_Output/'
n_filters = [128, 64, 32, 16]  # for a four layer CNN generator-critic
print (keras.__version__)


def train_for_epochs(image_set, generator, discriminator_model, gan, losses, n_train, batch_size=128,
                     n_epochs=100, save_every=10, save_filename_prefix=None):
    label_smooth = 0.1  # label smoothing factor
    dim_noise = generator.layers[0].input_shape[1]
    # n_ch = discriminator_model.layers[-2].input_shape[3]
    n_ch = 1
    for ie in range(n_epochs):
        print ('epoch: %d' % (ie + 1))
        idx_randperm = np.random.permutation(n_train)
        n_batches = int(np.floor(n_train / (batch_size * n_critic)))
        progbar = generic_utils.Progbar(n_batches * batch_size)
        for ib in range(n_batches):
            idx_batch = idx_randperm[range(ib * batch_size, ib * batch_size + batch_size)]
            X_real = image_set[idx_batch]
            X_real = np.expand_dims(X_real, axis=3)

            # sampled data
            noise_train, label_train = get_noise(dim_noise, batch_size=batch_size)

            positive_y = np.ones((batch_size, 1), dtype=np.float32)
            negative_y = -positive_y
            dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

            # train discriminator
            set_trainable(discriminator_model, True)
            set_trainable(generator, False)
            # train real batch

            d_loss = []
            for _ in range(n_critic):
                idx_batch = idx_randperm[range(ib * batch_size, ib * batch_size + batch_size)]
                X_real = image_set[idx_batch]
                X_real = np.expand_dims(X_real, axis=3)

                # train real batch
                d_loss_real = discriminator_model.train_on_batch([X_real, noise_train, label_train],
                                                                 [positive_y, negative_y, dummy_y])

                d_loss.append(d_loss_real)

            d_loss = np.asarray(d_loss, dtype='float32')
            d_loss_mean = np.mean(d_loss, axis=1)
            losses['d'].append(d_loss_mean)

            # train generator and auxiliary
            set_trainable(discriminator_model, False)
            set_trainable(generator, True)
            # generator loss and auxiliary loss
            g_loss = gan.train_on_batch([X_real, noise_train, label_train],
                                        [positive_y, label_train])
            losses['g'].append(g_loss)
            set_trainable(discriminator_model, True)

            # update progress bar
            progbar.add(batch_size, values=[("G loss", g_loss[1]), ("Q loss", g_loss[2]),
                                            ("D loss", d_loss[1]), ("D mse", d_loss[3])])

        # plot interim results
        if ((ie + 1) % save_every == 0) or (ie == n_epochs - 1):
            # display generated images channel by channel
            # generate_images(generator, output_dir, ie, n_class, batch_size=batch_size)
            for ic in range(n_ch):
                save_filename_image_gen = '%s_cat_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, ie + 1)
                plot_gen(generator, (10, 10), (15, 15), ic, True, save_filename_image_gen)

    # plot loss
    save_filename_loss = '%s_loss_epoch%d.pdf' % (save_filename_prefix, n_epochs)
    plot_loss(losses, True, save_filename_loss)


if __name__ == "__main__":
    losses = {'g': [], 'd': []}
    n_ch = 1
    dim_noise = 100
    n_epochs = 600
    batch_size = 50
    d_lr = 5e-4
    g_lr = 5e-4
    n_save_every = 1
    d_beta1 = 0.5
    d_beta2 = 0.9
    save_filename_prefix = '/home/SDash2/Desktop/MNIST_Output/infogan_noise%d_cat' % (dim_noise)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    x_train = (x_train.astype(np.float32) / 255.0)

    n_rows, n_cols = x_train.shape[1:]
    generator = build_generator(n_rows, n_cols, n_out_ch=n_ch, n_first_conv_ch=128, dim_noise=dim_noise,
                                dim_cat=n_class)
    plot_model(generator, to_file='%s_generator_model.pdf' % save_filename_prefix, show_shapes=True)
    discriminator, auxiliary = build_discriminator(n_rows, n_cols, n_in_ch=n_ch, n_last_conv_ch=128, lr=d_lr,
                                                   leaky_relu_alpha=0.2,
                                                   beta_1=0.5)
    plot_model(discriminator, to_file='%s_discriminator_model.pdf' % save_filename_prefix, show_shapes=True)
    plot_model(auxiliary, to_file='%s_auxiliary_model.pdf' % save_filename_prefix, show_shapes=True)
    # combined InfoGAN network
    discriminator_model, infogan = build_infogan(generator, discriminator, auxiliary,
                            g_lr, beta_1=d_beta1)
    plot_model(infogan, to_file='%s_model.pdf' % save_filename_prefix, show_shapes=True)
    gan_weights_file = '%s_weights_epoch%d_new%d_1.hdf' % (save_filename_prefix, n_epochs, n_class)
    if path.isfile(gan_weights_file):
        print ('loading InfoGAN model weights')
        infogan.load_weights(gan_weights_file)
    else:
        # train InfoGAN
        print ('training InfoGAN model')
        n_train = int(x_train.shape[0] / batch_size)
        train_for_epochs(x_train, generator,
                         discriminator_model, infogan, losses,
                         n_train=n_train,
                         batch_size=batch_size,
                         n_epochs=n_epochs,
                         save_every=n_save_every,
                         save_filename_prefix=save_filename_prefix)
        infogan.save_weights(gan_weights_file)

    for ic in range(1):
        save_filename_image_gen = '%s_cat_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, n_epochs)
        plot_gen(generator, (10, 10), (15, 15), ic, True, save_filename_image_gen)