# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:02:39 2018

Deep Attentional Multimodal Similarity Model score based on text and image features
@author: Shubham
"""
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import sys
from image_feature_extraction import *
from text_feature_extraction import *
