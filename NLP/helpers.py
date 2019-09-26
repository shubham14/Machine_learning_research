# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:18:52 2019

@author: Shubham
"""

from os.path import join as pjoin
import re
import numpy as np
import pandas as pd
from fastText import load_model

window_length = 200

BASE_DIR = r'C:\\Users\\Shubham\\Desktop\\Flow-Gans'

def mode(arg):
    def mode_wrapper(func):
        def func_wrapper():
            dir_path = pjoin(BASE_DIR, arg.strip() + '.csv')
            return func(dir_path)
        return func_wrapper
    return mode_wrapper

def clean(s):
    return [w.strip(',."!?:;()\'') for w in s]

def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s