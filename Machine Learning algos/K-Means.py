# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:48:11 2018

@author: Shubham
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

class KMeans:
    def __init__(self, train_data, tolerence, num, max_iterations=500):
        self.train_data = train_data
        self.tolerence = tolerence
        self.num = num
        self.centroids = []
        for i in range(num):
            self.centroids.append(data[i])
    
    # p and q are two vectors of n dimension
    def dist_metric(self, p, q):
        c = np.sum(np.power((a - b), 2))
        return np.sqrt(c)
        
    def fit(self):
        for _ in self.max_iterations:
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
            
            for data_points in data:
                dist = list(map(lambda x: self.dist_metric(data_points, x), self.centroids))
                class_data = dist.index(min(dist))
                self.classification[class_data].append(data_points)
                
            prev_centroids = dict(self.centroids)
            
            #update the centroids
            for i in range(len(self.centroids)):
                self.centroids[self.centroids[i]] = np.average(self.classification[self.centroids[i]], axis=1)
            
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break
    
    # predict new data points
    def predict(self, test_data):
        dist = list(map(lambda x: self.dist_metric(data_points, x), self.centroids))
        class_data = dist.index(min(dist))
        return class_data
    
    

            
            