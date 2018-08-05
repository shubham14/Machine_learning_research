# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 00:14:17 2018

@author: Shubham
"""

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

from os import path
from data_prepper import *
from network import *
import pickle

save_h5_filename = r"\C:\Users\Desktop\omniglot.hdf"
datapath = '/Users/Shubham/Desktop/data/Omniglot'

with open(os.path.join(datapath, "train.pickle"), "rb") as f:
    (Xtrain,c) = pickle.load(f)

with open(os.path.join(datapath, "val.pickle"), "rb") as f:
    (Xtest,cval) = pickle.load(f)

# training hyperparameters
evaluate_every = 7000
loss_every=300
batch_size = 32
N_way = 20
n_val = 550

# network parameters
n_filters=[64, 128, 128, 256]
n_rows, n_cols, n_ch =  105, 105, 1
fc_units = 1024
beta1 = 0.5 
beta2 = 0.9 
lr=1e-3
batch_size = 32
n_out_ch=1

if path.isfile(save_h5_filename):
    siamese_net.load_weights(save_h5_filename)
else:
    siamese_net = SiameseNetWork(n_filters, n_rows, n_cols,
                                 n_ch, fc_units, beta1, beta2,
                                 lr ,n_out_ch)
    diffNet = siamese_net.diff_net()
    best = 76.0
    data_prep = Siamese_Loader(Xtrain, Xtest)
    for i in range(900000):
        (inputs,targets)=data_prep.get_batch(batch_size)
        loss = diffNet.train_on_batch([inputs[0], inputs[1]], targets)
        if i % evaluate_every == 0:
            val_acc = data_prep.test_oneshot(diffNet, N_way, n_val, verbose=True)
            if val_acc >= best:
                print("saving")
                diffNet.save(dataPath)
                best = val_acc
    
        if i % loss_every == 0:
            print("iteration {}, training loss: {:.2f},".format(i, loss))