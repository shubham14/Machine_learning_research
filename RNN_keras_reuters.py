# -*- coding: utf-8 -*-
"""
Created on Sun May 20 10:57:51 2018

@author: Shubham
"""

import keras
from keras.layers import SimpleRNN, Embedding, Dense
from keras.models import Sequential
import numpy as np
from keras.datasets import reuters
from keras.preprocessing import sequence
import sys
from keras.utils import np_utils

max_features = 10000
maxlen = 500
batch_size = 64

print('Loading Data...')
(input_train, y_train), (input_test, y_test) = reuters.load_data(num_words=max_features)

                                                
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

y_train = np_utils.to_categorical(y_train, 46)
y_test = np_utils.to_categorical(y_test, 46)

model = Sequential()
model.add(Embedding(max_features, 64))
model.add(SimpleRNN(64, return_sequences=True))
model.add(SimpleRNN(64, return_sequences=True))
model.add(SimpleRNN(64, return_sequences=True))
model.add(SimpleRNN(64))
model.add(Dense(46, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.3)
