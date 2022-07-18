# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:39:03 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# from generateRawData import generateRawData


'''
DESCRIPTION: randomly selects unique column values and replaces with new value
from same column

INPUTS:
data = dataframe
nrows = number of rows that will be created in augmentation
nvalues = number of values being swapped, limited to column length
'''

def betterRandSwap(data, nrows, nvalues):
    # Creates empty dataframe to store augmented rows
    augmentedDf = pd.DataFrame()
    
    # Copies nrows from original data and appends to augmentedDf
    for i in range(nrows):
        augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
        
    # Drops labels column from augmentedDF
    augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
    
    # Picks UNIQUE column indexes to swap
    columnIndexSwaps = random.sample(range(0, data.shape[1]-1), nvalues)

    # Swaps augmentedDf column value from same column in data
    for i in range(augmentedDf.shape[0]):
        for col in columnIndexSwaps:
            randValue = data.iloc[random.randint(0, data.shape[0]-1), col]
            
            augmentedDf.iloc[i, col] = randValue
        
    # Combines both data and augmentedDf into one dataframe
    augmentedDf = pd.concat([data, augmentedDf], axis=0, ignore_index=True)
    
    return augmentedDf
