# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:37:42 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import random



'''
DESCRIPTION: randomly selects unique column values and replaces with 
Gaussian noise centered at zero with designated noise as std.

INPUTS:
data = dataframe
nrows = number of rows that will be created in augmentation
nvalues = number of values being swapped, limited to column length
noise = amount of noise, aka std.
'''

def betterGausNoise(data, nrows, nvalues, noise):
    
    # Creates empty dataframe to hold augmented rows
    augmentedDf = pd.DataFrame()
    
    # Selects random rows from data and appends to augmentedDf
    for i in range(nrows):
        augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
    
    # Drops label column of augmentedDf
    augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
    
    # Selects random unique column index 
    randCols = random.sample(range(0, data.shape[1]-1), nvalues)
    
    # Applies Gaussian noise to randCols values stored in array
    for i in range(augmentedDf.shape[0]):
        for cols in randCols:
            augmentedDf.iloc[i, cols] += np.random.normal(0, noise)
        
    
    # Combines both data and augmentedDf for full augmented dataframe
    augmentedDf = pd.concat([data, augmentedDf], ignore_index=True)
        
    
    return augmentedDf
    