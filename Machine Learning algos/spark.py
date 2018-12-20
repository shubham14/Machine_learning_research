# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:46:17 2018

@author: Shubham
"""

import pyspark
import numpy as np
import random

num_sample = 100000000
sc = pyspark.SparkContext(appName="Pi")

def inside(p):
    x, y = random.random(), random.random()
    return x*x + y*y < 1


count = sc.parallelize(range(0, num_sample)).filter(inside).count()

pi = 4 * count / num_sample
print(pi)
sc.stop()