# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:24:50 2018

@author: Shubham
"""

import numpy as np
import copy
from scipy.ndimage import imread
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

nrun = 20
fname_label = 'class_labels.txt'

def classification_run(folder, f_load, f_cost, ftype='cost'):
    assert ((ftype=='cost') | (ftype=='score'))
    with open(folder + '\\' + fname_label) as f:
        content = f.read().splitlines()
    pairs = [line.split() for line in content]
    test_files = [pair[0] for pair in pairs]
    train_files = [pair[1] for pair in pairs]
    answers_files = copy.copy(train_files)
    test_files.sort()
    train_files.sort()
    ntrain = len(train_files)
    ntest = len(test_files)
    
    train_items = [f_load(f) for f in train_files]
    test_items = [f_load(f) for f in test_files]
    
    # compute cost matrix
    costM = np.zeros((ntest, ntrain), float)
    for i in range(ntest):
        for c in range(ntrain):
            costM[i, c] = f_cost(test_items[i], train_items[c])
    if ftype == 'cost':
        yhat = np.argmin(costM, axis=1)
    elif ftype == 'score':
        yhat = np.argmin(costM, axis=1)
    else:
        assert False
    
    correct = 0.0
    for i in range(ntest):
        if train_files[YHAT[i]] == answers_files[i]:
            correct += 1.0
    pcorrect = 100 * correct / ntest
    perror = 100 - pcorrect
    return perror

def ModHausdorfDistance(itemA, itemB):
    D = cdist(itemA, itemB)
    mindist_A = D.min(axis=1)
    mindist_B = D.min(axis=0)
    mean_A = np.mean(mindist_A)
    mean_B = np.mean(mindist_B)
    return max(mean_A, mean_B)

def mean_axis(A):
    return A - np.mean(A)

def LoadImgAsPoints(fn):
    I = plt.imread(fn, flatten=True)
    I = np.array(I, dtype=bool)
    I = np.logical_not(I)
    (r, c) = I.nonzero()
    D = np.array([r, c])
    D = D.astype(float)
    D = np.apply_along_axis(mean_axis, A, 0)
    return D    
    
if __name__ == "__main__":
    perror = np.zeros(nrun)
    for r in range(1,nrun+1):
        rs = str(r)
        if len(rs)==1:
            rs = '0' + rs
        perror[r-1] = classification_run('run'+rs, LoadImgAsPoints, ModHausdorffDistance, 'cost')
        print (" run: " + str(r) + " (error: " + str(	perror[r-1] ) + "%)")		
        total = np.mean(perror)
    print ("average error: " + str(total) + "%")