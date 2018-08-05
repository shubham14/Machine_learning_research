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
import keras
from keras.datasets import *
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, Activation, Lambda, GlobalAveragePooling2D
from keras.layers.convolutional import Conv1D, Conv2D, Conv2DTranspose, Conv1DTranspose
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

# parameters in the paper
n_class = 10
LAMBDA = 10
LAMBDA2 = 2
drop_ratio = 0.2
EPSILON = 1e-8
n_critic = 5
output_dir = '/home/SDash2/Desktop/MNIST_Output/'
n_filters = [128, 64, 32, 16]  # for a four layer CNN generator-critic
print (keras.__version__)


def wasser_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def mean_loss(y_true, y_pred):
    return K.mean(y_pred)


def disc_mutual_info_loss(c_disc, aux_dist):
    """
    Mutual Information lower bound loss for discrete distribution.
    """
    reg_disc_dim = aux_dist.get_shape().as_list()[-1]
    cross_ent = - K.mean(K.sum(K.log(aux_dist + EPSILON) * c_disc, axis=1))
    ent = - K.mean(K.sum(K.log(1./reg_disc_dim + EPSILON) * c_disc, axis=1))
    return ent + cross_ent


def mean_loss(y_true, y_pred):
    return K.mean(y_pred)

def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for display."""
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


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


# Loss layer defined as classes
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class WeightedAverage(_Merge):
    def _merge_function(self, inputs):
        return (inputs[0]) + (0.1 * inputs[1])


# freeze weights in the discriminator for stacked training
def set_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def CT_Loss(inputs):
    a = inputs[0]
    b = inputs[1]
    soft_a = inputs[2]
    soft_b = inputs[3]
    mse_first = tf.reduce_mean(tf.squared_difference(a, b))
    mse_second = tf.reduce_mean(tf.squared_difference(soft_a, soft_b))

#    weighted_sum = WeightedAverage()([mse_first, mse_second])
    weighted_sum = tf.add(mse_first,  tf.multiply(0.1 , mse_second))
    CT_output = tf.maximum(0., weighted_sum)
    return CT_output
    

def build_generator(n_rows, n_cols, n_out_ch=1, n_first_conv_ch=128, dim_noise=100, dim_cat=n_class):
    g_in_noise = Input(shape=(dim_noise,))
    g_in_cat = Input(shape=(dim_cat,))
    g_in = concatenate([g_in_noise, g_in_cat])
    x = Dense(n_first_conv_ch * int(n_rows/4) * int(n_cols/4))(g_in)
    x = Reshape((int(n_rows/4), int(n_cols/4), n_first_conv_ch))(x)
    x = BatchNormalization(axis=-1, momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1DTranspose(int(n_first_conv_ch / 4), 9, strides=4)(x)
    x = BatchNormalization(axis=-1, momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1DTranspose(int(n_first_conv_ch / 16), 9, strides=4)(x)
    x = BatchNormalization(axis=-1, momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1DTranspose(int(n_first_conv_ch / 64), 9, strides=4)(x)
    x = BatchNormalization(axis=-1, momentum=0.8)(x)
    g_out = Conv1DTranspose(n_out_ch, 9, strides=1, padding='same', activation='sigmoid')(x)
    generator = Model([g_in_noise, g_in_cat], g_out, name='generator')
    print ('Summary of Generator (for InfoGAN)')
    generator.summary()
    return generator


def build_discriminator(n_rows, n_cols, n_in_ch=1, n_last_conv_ch=128, leaky_relu_alpha=0.2, lr=1e-4, beta_1=0.5):
    d_opt = Adam(lr=lr, beta_1=beta_1)
    d_in = Input(shape=(n_rows, n_cols, n_in_ch))

    x = Conv1D(int(n_last_conv_ch / 4), 9, strides= 4, padding='same')(d_in)
    x = LeakyReLU()(x)
    x = Dropout(rate=drop_ratio)(x)
    x = Conv1D(int(n_last_conv_ch), 9, strides=4)(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=drop_ratio)(x)
    x = Conv1D(n_last_conv_ch, 9, strides=4, padding='same')(x)
    x = LeakyReLU()(x)


    y = Conv1D(int(n_last_conv_ch / 2), 9, strides=4, padding='same')(d_in)
    y = LeakyReLU()(y)
    y = Dropout(rate=drop_ratio)(y)
    y = Conv1D(int(n_last_conv_ch), 9, strides=4)(y)
    y = LeakyReLU()(y)
    y = Dropout(rate=drop_ratio)(y)
    y = Conv1D(n_last_conv_ch, 9, strides=4, padding='same')(y)
    y = LeakyReLU()(y)

    x_flat = Flatten()(x)
    d_out_base = Dense(1024)(x_flat)
    d_out_disc = LeakyReLU(alpha=leaky_relu_alpha)(d_out_base)
    softmax_x = Dense(n_class, activation='softmax')(x_flat)


    y_flat = Flatten()(y)
    d_out_base_y = Dense(1024)(y_flat)
    d_out_disc_y = LeakyReLU(alpha=leaky_relu_alpha)(d_out_base_y)
    softmax_y = Dense(n_class, activation='softmax')(y_flat)

#    mse_first = MSELayer(output_dim=(n_class, 1)).call([x_flat, y_flat])
#    mse_second = MSELayer(output_dim=(n_class, 1)).call([softmax_x, softmax_y])
    CT_output = Lambda(CT_Loss)([x_flat, y_flat, softmax_x, softmax_y])
    
    # real probability output
    d_out_real = Dense(1)(d_out_disc)

    # categorical output
    x = Dense(1024)(d_out_base)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    d_out_cat = Dense(n_class, activation='softmax')(x)

    discriminator = Model(d_in, [d_out_real, CT_output], name='discriminator')
    # discriminator.compile(optimizer=d_opt, loss='mse')
    print ('Summary of Discriminator (for InfoGAN)')
    discriminator.summary()

    auxiliary = Model(d_in, d_out_cat, name='auxiliary')
    # auxiliary.compile(optimizer=d_opt, loss='categorical_crossentropy', metrics=['mean_squared_error'])
    print ('Summary of Auxiliary (for InfoGAN)')
    auxiliary.summary()

    return discriminator, auxiliary


def build_infogan(generator, discriminator, auxiliary, g_lr=1e-4, beta_1=0.5, beta_2=0.9):
    d_opt = Adam(lr=d_lr, beta_1=beta_1, beta_2=beta_2)
    gan_opt = Adam(lr=g_lr, beta_1=beta_1, beta_2=beta_2)
    set_trainable(discriminator, False)
    real_samples = Input(shape=discriminator.layers[0].input_shape[1:])
    generator_input_for_discriminator = Input(shape=generator.layers[0].input_shape[1:])
    gen_cat_inp_disc = Input(shape=generator.layers[1].input_shape[1:])
    gen_out = generator([generator_input_for_discriminator, gen_cat_inp_disc])
    dis_gen, _ = discriminator(gen_out)

    gan_out_cat = auxiliary(gen_out)
    gan = Model([real_samples, generator_input_for_discriminator, gen_cat_inp_disc],
                [dis_gen, gan_out_cat])
    gan.compile(optimizer=gan_opt, loss=[wasser_loss,
                                         'categorical_crossentropy'],
                loss_weights=[1, 1])

    set_trainable(discriminator, True)
    set_trainable(generator, False)

    averaged_samples = RandomWeightedAverage()([real_samples, gen_out])
    averaged_samples_out, CT_output = discriminator(averaged_samples)

    dis_real, CT_out = discriminator(real_samples)
    dis_gen, _ = discriminator(gen_out)
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=LAMBDA)
    partial_gp_loss.__name__ = 'gradient penalty'

    discriminator_model = Model(inputs=[real_samples,
                                        generator_input_for_discriminator,
                                        gen_cat_inp_disc],
                                outputs=[dis_real, dis_gen,
                                         averaged_samples_out, CT_out])
    discriminator_model.compile(optimizer=d_opt, loss=[wasser_loss,
                                                       wasser_loss,
                                                       partial_gp_loss,
                                                       mean_loss],
                                loss_weights=[1, 1, 1, LAMBDA2])

    gan.summary()
    return discriminator_model, gan


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
                                                                 [positive_y, negative_y, dummy_y, dummy_y])

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
    dim_noise = 100
    n_epochs = 600
    batch_size = 50
    d_lr = 5e-4
    g_lr = 5e-4
    n_save_every = 1
    d_beta1 = 0.5
    d_beta2 = 0.9
    save_filename_prefix = '/Users/Shubham/Desktop/infogan_noise%d_cat' % (dim_noise)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    x_train = (x_train.astype(np.float32) / 255.0)

    n_rows, n_cols = x_train.shape[1:]
    n_ch = 1
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
        save_filename_image_gen = '%s_cifar10_cat_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, n_epochs)
        plot_gen(generator, (10, 10), (15, 15), ic, True, save_filename_image_gen)