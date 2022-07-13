# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:11:41 2022

@author: jeffr
"""
import numpy as np
import pandas as pd
import random

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
    