# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 20:30:13 2018

@author: Shubham
"""

import torch
import torch.nn as nn
import numpy as np
import random

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]
    