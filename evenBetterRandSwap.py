# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:39:03 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import random

from generateRawData import generateRawData

data = generateRawData(10, 3, 0.5, 'gaussian')

def betterRandSwap(df, nrows, nvalues):
    augmentedDf = pd.DataFrame()
    
    rowIndexCopies = random.sample(range(0, df.shape[0]-1), nrows)
    
    
    for row in rowIndexCopies:
        augmentedDf = pd.concat([augmentedDf, df.iloc[[row]]], ignore_index=True)
    
    augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
    
    
    columnIndexSwaps = random.sample(range(0, df.shape[1]-1), nvalues)
    
    # for col in columnIndexSwaps:
        
    
    return augmentedDf

aug = betterRandSwap(data, 5, 3)