# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 22:53:40 2018

@author: Shubham
"""

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

class loss():
    
    def __init__(self):
        pass
    
    def wasser_loss(self, y_true, y_pred):
        return K.mean(y_pred * y_pred)
    
    
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples, gradient_penalty_weight):
        grad = K.gradients(y_pred, averaged_samples)[0]
        grad_sqr = K.square(grad)
        grad_sqr_sum = K.sum(grad_sqr, axis=np.arange(1, len(grad_sqr.shape)))
        grad_l2_norm = K.sqrt(K.abs(grad_sqr_sum))
        grad_penalty = gradient_penalty_weight * K.square(1 - grad_l2_norm)
        return K.mean(grad_penalty)


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