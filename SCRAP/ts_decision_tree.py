#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:11:02 2022

@author: jasonwhite
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import random
import sklearn.metrics as skm
from sklearn.datasets import make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sea
from pyts.classification import KNeighborsClassifier
from pyts.datasets import load_gunpoint
from pyts.classification import TimeSeriesForest
from fakeSuper import applyAugmentationMethod


def ts_dtree(dataset, target, split):
    
    dfdrop = dataset.drop(columns = dataset.shape[1] - 1)
   
    X = dfdrop
    y = dataset[dataset.shape[1] - 1]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = split, shuffle = False,  stratify = None) 
    
    
    clf = TimeSeriesForest(random_state=43)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    for i in range(split, dataset.shape[0]):
        dataset.loc[i, target] = y_pred[i - split]
        
    return dataset

