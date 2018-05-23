import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import keras
from keras.layers import *
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model
from keras.datasets import mnist
from keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
import argparse
import os

# parameters
intermediate_dim = 1024
latent_dim = 45

# reparameterization in isotropic Gaussian 
def sample_z(args):
	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]

	# Gaussian noise ~ N(0,1)
	eps = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var) * eps

class Model():
	
	def load_proc_data():
		(x_tr, y_tr), (x_ts, y_ts) = mnist.load_data()
		image_size = x_tr.shape[1]
		org_dim = image_size * image_size
		x_tr = np.reshape(x_tr, [-1, org_dim]).astype('float32') / 255
		x_ts = np.reshape(x_ts, [-1, org_dim]).astype('float32') / 255
		return (x_tr, y_tr), (x_ts, y_ts)

	def build_vae(input_shape, output_shape):
		inp = Input(shape=(input_shape,), activation='relu')
		x = Dense(intermediate_dim, activation='relu')(inp)
		z_mean = Dense(latent_dim, name='z_mean')(x)
		z_log_var = Dense(latent_dim, activation='relu')(x)
		z = Lambda(sample_z, output_shape=(latent_dim,))([z_mean, z_log_var])

		# encoder
		encoder = Model(inp, [z_mean, z_log_var])
		encoder.summary()

		# decoder
		latent_input = Input(shape=(latent_dim,))
		x = Dense(intermediate_dim, activation='relu')(latent_input)
		outputs = Dense(org_dim, activation='sigmoid')(x)

