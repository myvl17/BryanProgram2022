# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:39:03 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from generateRawData import generateRawData

def betterRandSwap(data, nrows, nvalues):
    augmentedDf = pd.DataFrame()
    
    for i in range(nrows):
        augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
        
    # augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
    
    # TESTING PURPOSES
    for i in range(augmentedDf.shape[0]):
        augmentedDf.iloc[i, augmentedDf.shape[1]-1] = 2
    
    columnIndexSwaps = random.sample(range(0, data.shape[1]-1), nvalues)

    
    for i in range(augmentedDf.shape[0]):
        for col in columnIndexSwaps:
            randValue = data.iloc[random.randint(0, data.shape[0]-1), col]
            
            augmentedDf.iloc[i, col] = randValue
        
    augmentedDf = pd.concat([data, augmentedDf], axis=0, ignore_index=True)
    
    return augmentedDf
