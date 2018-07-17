# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:18:56 2018

@author: Shubham
"""

import os
import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

