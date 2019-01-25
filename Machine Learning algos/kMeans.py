# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:04:03 2019

@author: Shubham
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

class KMeans:
    def __init__(self, tol=0.002, max_iter=1000, K=2):
        self.tol = tol
        self.K = K
        self.max_iter = max_iter
        
    def fit(self, data):
        self.centroids = {}
        for i in range(self.K):
            self.centroids[i] = self.data[i]
        
        iter_count = 0
        t = 1234
        self.classifications = {}
        while iter_count < self.max_iter or t > self.tol:
            for i in range(self.K):
                self.classifications[i] = []
            
            for feature in data:
                dist = [np.linalg.norm(feature - self.centroids[centroid]) for centroid in self.centroids]
                class1 = dist.index(min(dist))
                self.classifications[class1].append(feature)
            
            prev_centroids = dict(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            
                        optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break
    
    def predict(self, data):
        dist = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        class1 = dist.index(min(dist))
        return class1
            
            
        