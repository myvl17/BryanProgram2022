# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:11:41 2022

@author: jeffr
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from generateRawData import generateRawData

from superFunction import logReg
from columnNumberTesting import runClassifier

data = generateRawData(10, 5, 0, 'gaussian')
print(runClassifier(data, 'SVM', 'f1'))


def modifiedGausNoise(df, nrows, nvalues):
    augmentedDf = pd.DataFrame()
    
    randCols = random.sample(range(0, df.shape[1]-1), nvalues)
    
    for i in range(nrows):
        augmentedDf = pd.concat([augmentedDf, df.iloc[[random.randint(0, df.shape[0]-1)]]], ignore_index=True)
        
    augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
    
        
    for col in randCols:
        for i in range(augmentedDf.shape[0]):
            mean = augmentedDf[col].mean()
            stDev = augmentedDf[col].std()
            
            augmentedDf.iloc[i, col] =  np.random.normal(mean, stDev)
        
        
    augmentedDf = pd.concat([df, augmentedDf], axis=0, ignore_index=True)
    
    return augmentedDf
    
    
    
aug = modifiedGausNoise(data, 10, 5)

log = logReg(aug, feature_cols = np.arange(0, aug.shape[1]-1), target=aug.shape[1]-1, split=data.shape[0]-1)

acc = runClassifier(log, 'SVM', 'f1')