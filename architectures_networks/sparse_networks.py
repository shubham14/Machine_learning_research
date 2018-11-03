# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:57:44 2018

@author: Shubham
"""

from keras import backend as K
from keras.engine.topology import Layer
import keras 
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, load_model, Sequential
from keras.utils import to_categorical, plot_model, np_utils
from tensorflow.python.keras.layers import Lambda
from keras import layers
import tensorflow as tf
from keras.engine import InputSpec
from keras.initializers import Initializer, VarianceScaling, _compute_fans


class AdjacencyInitializer(Initializer):
    def __init__(self, drop_prob=0):
        self.drop_prob = drop_prob

    def __call__(self, shape, dtype=None):
        mat = np.random.random_sample(shape) > self.drop_prob
        mat = mat.astype(int)
        return mat

    def get_config(self):
        return {'adjacency_mat': self.adjacency_mat}


class SparseLayer(Dense):
    def __init__(self,
                 drop_prob=0,
                 units=0,
                 *args,
                 **kwargs):
        self.drop_prob=drop_prob
        self.units = units
        super().__init__(units=units, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(
                input_dim,
                self.units),
            initializer=AdjacencyInitializer(
                    self.drop_prob
                ),
            name='adjacency_matrix',
            trainable=False)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = self.kernel * inputs
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def count_params(self):
        num_weights = 0
        if self.use_bias:
            num_weights += self.units
        num_weights += np.sum(self.kernel)
        return num_weights
        
    def get_config(self):
        config = {
            'adjacency_mat': self.kernel.tolist()
        }
        base_config = super().get_config()
        base_config.pop('units', None)
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        adjacency_mat_as_list = config['adjacency_mat']
        config['adjacency_mat'] = np.array(adjacency_mat_as_list)
        return cls(**config)

class Data:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def dataLoader(self):
        dataset = pd.read_csv(self.data_path)
        X = dataset.iloc[:, 0:4].values
        y = dataset.iloc[:, 4].values
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=0)
    
        return X_train, X_test, y_train, y_test
    
if __name__ == "__main__":
    data_path = r'C:\Users\Shubham\Anaconda31\Lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\data\iris.csv'
    data = Data(data_path)
    X_train, X_test, y_train, y_test = data.dataLoader()
    m, n =  X_train.shape
    model = Sequential()
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(SparseLayer(drop_prob=0.2, units=10))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X_train, y_train, epochs=150, batch_size=10)