''' Python implementation for Variational Autoencoders 
SDash, May 2018
'''
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
import os

# parameters for the model
intermediate_dim = 1024
latent_dim = 45
alpha = 1.

# reparameterization in isotropic Gaussian 
def sample_z(args):
	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]

	# Gaussian noise ~ N(0,1)
	eps = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var) * eps

class Model():
	
	def __init__(self, image_size, alpha, intermediate_dim, latent_dim):
		self.org_dim = image_size * image_size
		self.alpha = alpha
		self.intermediate_dim = intermediate_dim
		self.latent_dim = latent_dim

	# load and process the MNIST data
	def load_proc_data(self, org_dim):
		(x_tr, y_tr), (x_ts, y_ts) = mnist.load_data()
		image_size = x_tr.shape[1]
		x_tr = np.reshape(x_tr, [-1, org_dim]).astype('float32') / 255
		x_ts = np.reshape(x_ts, [-1, org_dim]).astype('float32') / 255
		return (x_tr, y_tr), (x_ts, y_ts)

	def Loss(self, z_log_var, z_mean, x, x_decoded_mean):
		kl_loss = 1. + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
		recon_loss = self.alpha * self.org_dim * objectives.categorical_crossentropy(x, x_decoded_mean)


	# build the Variational Autoencoders
	def build_vae(self, input_shape, output_shape):
		inp = Input(shape=(input_shape,), activation='relu')
		x = Dense(self.intermediate_dim, activation='relu')(inp)
		z_mean = Dense(self.latent_dim, name='z_mean')(x)
		z_log_var = Dense(self.latent_dim, activation='relu')(x)
		z = Lambda(sample_z, output_shape=(latent_dim,))([z_mean, z_log_var])

		# general encoder-decoder architecture subject to change

		# encoder
		encoder = Model(inp, [z_mean, z_log_var])
		encoder.summary()

		# decoder
		latent_input = Input(shape=(latent_dim,))
		x = Dense(intermediate_dim, activation='relu')(latent_input)
		outputs = Dense(org_dim, activation='sigmoid')(x)
		decoder = Model(latent_input, outputs)
		decoder.summary()

		outputs = decoder(encoder(inp)[2])
		vae = Input(inp, outputs)
		return vae

# main function
if __name__ == "__main__":

	# instatiate the Class and load the data
	model = Model(img_dim, alpha, intermediate_dim, latent_dim)
	x_tr, y_tr, x_ts, y_ts = model.load_proc_data()
	
	# building the variational autoencoder and fitting the model
	vae = build_vae(org_dim, org_dim)
	vae.fit(x_tr, y_tr)

	# predictions on the test data
	y_pred = vae.predict(x_ts)