# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:30:46 2018

You might not need the sigmoid layer in the discriminator
Try to compile the discriminator in the function
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
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, Activation, Lambda, GlobalAveragePooling2D
from keras.layers.convolutional import Conv1D, Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.layers.merge import _Merge
from keras.utils import \
    plot_model  # for model visualization, need to install Graphviz (.msi for Windows), pydot (pip install), graphviz (pip install) and set PATH for Graphviz
from keras.optimizers import *
from keras.engine.topology import Layer
from keras import backend as K
from functools import partial
import keras.losses
from keras.layers.core import *
import tensorflow as tf

K.set_image_data_format('channels_last')
K.set_image_dim_ordering('tf')

n_class = 10
LAMBDA = 10
EPSILON = 1e-8
n_filters = [128, 64, 32, 16]  # for a four layer CNN generator-critic
print (keras.__version__)


def wasser_loss(y_true, y_pred):
    return K.mean(y_pred * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    grad = K.gradients(y_pred, averaged_samples)[0]
    grad_sqr = K.square(grad)
    grad_sqr_sum = K.sum(grad_sqr, axis=np.arange(1, len(grad_sqr.shape)))
    grad_l2_norm = K.sqrt(K.abs(grad_sqr_sum))
    grad_penalty = gradient_penalty_weight * K.square(1 - grad_l2_norm)
    return K.mean(grad_penalty)


def plot_loss(losses, save_file=False, file_name=None):
    plt.figure(figsize=(10, 10))
    g_loss = np.array(losses['g'])
    d_loss = np.array(losses['d'])
    plt.plot(g_loss[:, 1], label='G loss')
    plt.plot(g_loss[:, 2], label='Q loss')
    plt.plot(g_loss[:, 1], label='D loss')
    plt.plot(g_loss[:, 2], label='D mse')
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


# Loss layers defined as classes
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class Subtract(_Merge):
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = output - inputs[i]
        return output


class GradNorm(Layer):
    def __init__(self, **kwargs):
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(GradNorm, self).build(input_shapes)

    def call(self, inputs):
        target, wrt = inputs
        grads = K.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


# freeze weights in the discriminator for stacked training
def set_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def build_generator(n_rows, n_cols, n_out_ch=1, n_first_conv_ch=128, dim_noise=100, dim_cat=n_class):
    g_in_noise = Input(shape=(dim_noise,))
    g_in_cat = Input(shape=(dim_cat,))
    g_in = concatenate([g_in_noise, g_in_cat])
    x = Dense(n_first_conv_ch * int(n_rows) * int(n_cols))(g_in)
    x = Reshape((int(n_rows), int(n_cols), n_first_conv_ch))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(int(n_first_conv_ch / 2), (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(int(n_first_conv_ch / 4), (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(int(n_first_conv_ch / 8), (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    g_out = Conv2DTranspose(n_out_ch, (3, 3), strides=1, padding='same', activation='tanh')(x)
    generator = Model([g_in_noise, g_in_cat], g_out, name='generator')
    print ('Summary of Generator (for InfoGAN)')
    generator.summary()
    return generator


def build_discriminator(n_rows, n_cols, n_in_ch=1, n_last_conv_ch=128, leaky_relu_alpha=0.2, lr=1e-4, beta_1=0.5):
    d_opt = Adam(lr=lr, beta_1=beta_1)
    d_in = Input(shape=(n_rows, n_cols, n_in_ch))
    x = Conv2D(int(n_last_conv_ch / 8), (3, 3), strides=1, padding='same')(d_in)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(int(n_last_conv_ch / 4), (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(int(n_last_conv_ch / 2), (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(n_last_conv_ch, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha, name='d_feature')(x)
    d_out_base = Flatten()(x)

    # real probability output
    d_out_real = Dense(1)(d_out_base)

    # categorical output
    x = Dense(128)(d_out_base)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    d_out_cat = Dense(n_class, activation='sigmoid')(x)

    discriminator = Model(d_in, d_out_real, name='discriminator')
    discriminator.compile(optimizer=d_opt, loss='binary_crossentropy', metrics=['mean_squared_error'])
    print ('Summary of Discriminator (for InfoGAN)')
    discriminator.summary()

    auxiliary = Model(d_in, d_out_cat, name='auxiliary')
    auxiliary.compile(optimizer=d_opt, loss='categorical_crossentropy', metrics=['mean_squared_error'])
    print ('Summary of Auxiliary (for InfoGAN)')
    auxiliary.summary()

    return discriminator, auxiliary


def infogan_model_wrapper_new(generator, discriminator, auxiliary,
                              d_lr=2e-4, beta_1=0.5, beta_2=0.9):
    d_opt = Adam(lr=d_lr, beta_1=beta_1, beta_2=beta_2)
    set_trainable(discriminator, False)
    real_samples = Input(shape=discriminator.layers[0].input_shape[1:])
    generator_input_for_discriminator = Input(shape=generator.layers[0].input_shape[1:])
    gen_cat_inp_disc = Input(shape=generator.layers[1].input_shape[1:])
    gen_out = generator([generator_input_for_discriminator, gen_cat_inp_disc])

    averaged_samples = RandomWeightedAverage()([real_samples, gen_out])
    averaged_samples_out = discriminator(averaged_samples)
    sub = Subtract()([discriminator(gen_out), discriminator(real_samples)])
    norm = GradNorm()([averaged_samples_out, averaged_samples])
    #    sub = Subtract()([discriminator(gen_out), discriminator(real_samples)])
    #    norm = GradNorm()([averaged_samples_out, averaged_samples])
    #    prob = Dense(1, activation='sigmoid')(norm)
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


# Hard coded parameters
def build_infogan(generator, discriminator, discriminator_model,
                  auxiliary, g_lr=1e-4, beta_1=0.5, beta_2=0.9):
    gan_opt = Adam(lr=g_lr, beta_1=beta_1)
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


def train_for_epochs(image_set, generator, discriminator, discriminator_model, gan, losses, n_train, batch_size=128,
                     n_epochs=100,
                     save_every=10, save_filename_prefix=None):
    label_smooth = 0.1  # label smoothing factor
    dim_noise = generator.layers[0].input_shape[1]
    # n_ch = discriminator_model.layers[-2].input_shape[3]
    n_ch = 1
    for ie in range(n_epochs):
        print ('epoch: %d' % (ie + 1))
        idx_randperm = np.random.permutation(n_train)
        n_batches = int(np.floor(n_train / batch_size))
        progbar = generic_utils.Progbar(n_batches * batch_size)
        for ib in range(n_batches):
            # train discriminator
            set_trainable(discriminator, True)
            # train real batch
            positive_y = np.ones((batch_size, 1), dtype=np.float32)
            negative_y = -positive_y
            dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
            y_1 = np.ones((batch_size,))
            idx_batch = idx_randperm[range(ib * batch_size, ib * batch_size + batch_size)]
            noise_train, label_train = get_noise(dim_noise, batch_size=batch_size)  # generate noise and labels
            X_real = image_set[idx_batch]
            X_real = np.expand_dims(X_real, axis=3)
            #            dis_real_out = discriminator.predict(X_real)
            y_real = np.random.uniform(low=1 - label_smooth, high=1, size=(batch_size,))  # label smoothing
            X_fake = generator.predict([noise_train, label_train])
            c = np.random.random()
            aver_in = c * X_real + (1 - c) * X_fake
            aver_out = discriminator.predict(aver_in)
            #            aver_out = np.random.uniform(size=(batch_size,))
            #            X_fake = np.expand_dims(X_fake, axis=3)
            y_fake = np.random.uniform(low=0, high=label_smooth, size=(batch_size,))  # label smoothing
            #            dis_fake_out = discriminator.predict(X_fake)
            # train real batch
            d_loss_real = discriminator_model.train_on_batch([X_real, noise_train, label_train],
                                                             [positive_y, dummy_y])

            # train fake batch
            d_loss_fake = discriminator_model.train_on_batch([X_fake, noise_train, label_train],
                                                             [positive_y, dummy_y])

            # discriminator loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            losses['d'].append(d_loss)

            # train generator and auxiliary
            noise_train, label_train = get_noise(dim_noise, batch_size=batch_size)
            set_trainable(discriminator_model, False)
            # generator loss and auxiliary loss
            g_loss = gan.train_on_batch([X_real, noise_train, label_train],
                                        [positive_y, dummy_y, label_train])
            losses['g'].append(g_loss)

            # update progress bar
            progbar.add(batch_size, values=[("G loss", g_loss[1]), ("Q loss", g_loss[3]),
                                            ("D loss", d_loss[1]), ("D mse", d_loss[2])])

        # plot interim results
        if ((ie + 1) % save_every == 0) or (ie == n_epochs - 1):
            # display generated images channel by channel
            for ic in range(n_ch):
                save_filename_image_gen = '%s_cat_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, ie + 1)
                plot_gen(generator, (10, 10), (15, 15), ic, True, save_filename_image_gen)

    # plot loss
    save_filename_loss = '%s_loss_epoch%d.pdf' % (save_filename_prefix, n_epochs)
    plot_loss(losses, True, save_filename_loss)


if __name__ == "__main__":
    losses = {'g': [], 'd': []}
    n_ch = 1
    dim_noise = 50
    n_epochs = 200
    batch_size = 120
    d_lr = 2e-4
    g_lr = 1e-3
    n_save_every = 1
    d_beta1 = 0.5
    d_beta2 = 0.9
    save_filename_prefix = 'infogan_noise%d_cat' % (dim_noise)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
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
    generator_model, discriminator_model, auxiliary_infogan = infogan_model_wrapper_new(generator,
                                                                                        discriminator,
                                                                                        auxiliary,
                                                                                        d_lr=d_lr,
                                                                                        beta_1=d_beta1,
                                                                                        beta_2=d_beta2)
    infogan = build_infogan(generator_model, discriminator, discriminator_model, auxiliary_infogan,
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
        train_for_epochs(x_train, generator_model, discriminator,
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