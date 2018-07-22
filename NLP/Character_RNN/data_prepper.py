# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 18:38:05 2018
Contains data preprocessing functions on a charecter level
@author: Shubham
"""

from __future__ import unicode_literals, print_function, division
import glob
import os
from io import open
import unicodedata
import string
import torch

l = glob.glob('/Users/Shubham/Desktop/Machine_learning_research/data/Char_RNN/names/*.txt')

letters = string.ascii_letters + ".,;'-"
n_letters = len(letters)

def unicodeToAscii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in letters)
    
print (unicodeToAscii('Ślusàrski'))

category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in l:
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

def letter_to_ind(letter):
    return letters.find(letter)

def letter_to_tensor(letter):
    t = torch.zeros(1, len(letters))
    t[0][letter_to_ind(letter)] = 1
    return t

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, len(letters))
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_ind(letter)] = 1
    return tensor
