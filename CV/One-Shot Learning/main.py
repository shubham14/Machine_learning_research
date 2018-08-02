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
from data_prepper import *
from network import *

save_h5_filename = "\C:\Users\Desktop\omniglot.hdf"

# training hyperparameters
evaluate_every = 7000
loss_every=300
batch_size = 32
N_way = 20
n_val = 550

# network parameters
n_filters=[64, 128, 128, 256]
n_rows, n_cols, n_ch = 
fc_units = 1024
beta1 = 0.5 
beta2 = 0.9 
lr=1e-3
n_out_ch=1,

if path.isfile(save_h5_filename):
    siamese_net.load_weights(save_h5_filename)
else:
    siamese_net = Siamese_net()
    best = 76.0
    for i in range(900000):
        (inputs,targets)=loader.get_batch(batch_size)
        loss=siamese_net.train_on_batch(inputs,targets)
        if i % evaluate_every == 0:
            val_acc = Data_Prepper.accuracy(siamese_net,N_way,n_val,verbose=True)
            if val_acc >= best:
                print("saving")
                siamese_net.save('PATH')
                best=val_acc
    
        if i % loss_every == 0:
            print("iteration {}, training loss: {:.2f},".format(i,loss))