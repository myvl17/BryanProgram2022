# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:46:47 2022

@author: jeffr
"""

import pandas as pd
import random


'''
DESCRIPTION: randomly selects unique column values and adds plus or minus
designated unit

INPUTS:
data = dataframe
nrows = number of rows that will be created in augmentation
nvalues = number of values being swapped, limited to column length
unit = amount being plus or minused
'''

def betterPmOne(data, nrows, nvalues, unit):
    
    # Creates empty dataframe to store augmented data
    augmentedDf = pd.DataFrame()
    
    # Randomly selects rows from data and appends to augmentedDf
    for i in range(nrows):
        augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
        
    # Drops labels column in augmentedDf
    augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
    
    # Selects nvalues amount of unique column indexes
    randCols = random.sample(range(0, data.shape[1]-1), nvalues)
    
    # Iterates through augmentedData and applies plus or minus to randCols indexes
    for i in range(augmentedDf.shape[0]):
        for col in randCols:
            if (random.randint(0, 1) == 0):
                augmentedDf.iloc[i, col] += unit
            else:
                augmentedDf.iloc[i, col] -= unit

    # Combines data and augmentedDf into finished augmented dataframe
    augmentedDf = pd.concat([data, augmentedDf], ignore_index=True)
    
    return augmentedDf


    
