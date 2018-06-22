# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 08:34:43 2018

@author: Shubham
"""

import argparse
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
from functools import partial
from net import Net
try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! Please install it (e.g. with pip install pillow)')
    exit()

batchSize = 64
TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


def tile_images(image_stack):
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def generate_images(generator_model, output_dir, epoch):
    test_image_stack = generator_model.predict(np.random.rand(10, 100))
    test_image_stack = (test_image_stack * 127.5) + 127.5
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output, mode='L')  # L specifies greyscale
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)

def data_load():
    # First we load the image data, reshape it and normalize it to the range [-1, 1]
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.concatenate((X_train, X_test), axis=0)
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    else:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    return X_train


def WGAN_model_wrapper(generator, discriminator, X_train):
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    generator_input = Input(shape=(100,))
    generator_layers = generator(generator_input)
    discriminator_layers_for_generator = discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
    # We use the Adam paramaters from Gulrajani et al.
    generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)
    
    # Now that the generator_model is compiled, we can make the discriminator layers trainable.
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    real_samples = Input(shape=X_train.shape[1:])
    generator_input_for_discriminator = Input(shape=(100,))
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)
    
    averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
    averaged_samples_out = discriminator(averaged_samples)
    
    
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error
    
    
    discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])
    
    discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                                loss=[wasserstein_loss,
                                      wasserstein_loss,
                                      partial_gp_loss])

   return generator_model, discrminator_model

# main function
if __name__ == "__main__":
    net = Net(100)
    positive_y = np.ones((batchSize, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((batchSize, 1), dtype=np.float32)
    generator = net.make_generator()
    discriminator = net.make_discriminator()
    
    # wrapper around the generator and discriminator
    generator_model, discriminator_model = WGAM_model_wrapper(generator, discriminator)
    for epoch in range(100):
        np.random.shuffle(X_train)
        print("Epoch: ", epoch)
        print("Number of batches: ", int(X_train.shape[0] // batchSize))
        discriminator_loss = []
        generator_loss = []
        minibatches_size = batchSize * TRAINING_RATIO
        for i in range(int(X_train.shape[0] // (batchSize * TRAINING_RATIO))):
            discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
            for j in range(TRAINING_RATIO):
                image_batch = discriminator_minibatches[j * batchSize:(j + 1) * BATCH_SIZE]
                noise = np.random.rand(batchSize, 100).astype(np.float32)
                discriminator_loss.append(discriminator_model.train_on_batch([image_batch, noise],
                                                                             [positive_y, negative_y, dummy_y]))
            generator_loss.append(generator_model.train_on_batch(np.random.rand(batchSize, 100), positive_y))
        # Still needs some code to display losses from the generator and discriminator, progress bars, etc.
        generate_images(generator, args.output_dir, epoch)