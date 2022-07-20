# -*- coding: utf-8 -*-
"""
Created on Tue Jul 5 15:00:37 2022

@author: cdiet
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random 
from betterPmOne import betterPmOne

x = [.25, .75, .4, .8, .9, 1, 1.1, .95, .925, .85]
y = [.3, .2, .75, .6, .75, .5, .75, .9, 1, 1.1]
 
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

df = pd.DataFrame({0:x, 1:y, 2:labels})

def correlationAugmentation(data, nrows, nvalues, unit):

    augmentedDf = pd.DataFrame()
    
    # Randomly selects rows from data and appends to augmentedDf
    for i in range(nrows):
        augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
        
    # Drops labels column in augmentedDf
    augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
    
    # Selects nvalues amount of unique column indexes
    # Make it so that we set a standard for what is ``considered to good correlation. i.e if correlation > 0.1
    randCols_1 = random.sample(range(0, data.shape[1]-1), nvalues)
    randCols_2 = random.sample(range(0, data.shape[1]-1), nvalues)
    
    for j in range(augmentedDf.shape[0]):
        for i in range(len(randCols_1)):
            #c  = augmentedDf.iloc[j, randCols_2[i]]/augmentedDf.iloc[j, randCols_1[i]]
            c = np.corrcoef(augmentedDf.iloc[:, randCols_2[i]], augmentedDf.iloc[:, randCols_1[i]])
            print(c)
            if random.randint(0, 1) == 1:
                augmentedDf.iloc[j, randCols_1[i]] += unit
            else:
                augmentedDf.iloc[j, randCols_1[i]] -= unit
            augmentedDf.iloc[j, randCols_2[i]] = c * augmentedDf.iloc[j, randCols_1[i]] + 0
    
    augmentedDf = pd.concat([data, augmentedDf], ignore_index= True)
    return augmentedDf
    
test = correlationAugmentation(df, 10, 2, unit= 1)