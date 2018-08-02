# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 00:14:17 2018

@author: Shubham
"""

from network import *
from data_prepper import *

# import keras libraries
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

evaluate_every = 7000
loss_every=300
batch_size = 32
N_way = 20
n_val = 550
siamese_net.load_weights("PATH")
best = 76.0
for i in range(900000):
    (inputs,targets)=loader.get_batch(batch_size)
    loss=siamese_net.train_on_batch(inputs,targets)
    if i % evaluate_every == 0:
        val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
        if val_acc >= best:
            print("saving")
            siamese_net.save('PATH')
            best=val_acc

    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i,loss))