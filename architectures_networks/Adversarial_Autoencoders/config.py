import torch

class Enc_cfg:
    '''
    Considering MNIST dataset as a default
    can change config according to different dataset
    '''
    cnn_filters = [16, 32, 64, 128]
    in_channels = 1
    drop_prob = 0.2

class Dec_cfg:
    '''
    Considering MNIST dataset as a default
    can change config according to different dataset
    '''
    in_channels = 128
    cnn_filters = [64, 32, 16, 1]
    drop_prob = 0.2

