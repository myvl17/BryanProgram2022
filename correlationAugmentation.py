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
z = [.17, .48, .32, .58, .63, 1.26, .98, .87, 1.19, 1.0]
 
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

df = pd.DataFrame({0:x, 1:y, 2:z, 3:labels})

# def correlationAugmentation(data, nrows, nvalues, unit):

#     augmentedDf = pd.DataFrame()
    
#     # Randomly selects rows from data and appends to augmentedDf
#     for i in range(nrows):
#         augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
        
#     # Drops labels column in augmentedDf
#     augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
    
#     # Selects nvalues amount of unique column indexes
#     # Make it so that we set a standard for what is ``considered to good correlation. i.e if correlation > 0.1
#     randCols_1 = random.sample(range(0, data.shape[1]-1), nvalues)
#     randCols_2 = random.sample(range(0, data.shape[1]-1), nvalues)
    
#     for j in range(augmentedDf.shape[0]):
#         for i in range(len(randCols_1)):
#             #c  = augmentedDf.iloc[j, randCols_2[i]]/augmentedDf.iloc[j, randCols_1[i]]
#             # c = np.corrcoef(augmentedDf.iloc[:, randCols_2[i]], augmentedDf.iloc[:, randCols_1[i]])
#             print(np.corrcoef(augmentedDf.iloc[:, randCols_2[i]], augmentedDf.iloc[:, randCols_1[i]]))
            
#             if random.randint(0, 1) == 1:
#                 augmentedDf.iloc[j, randCols_1[i]] += unit
#             else:
#                 augmentedDf.iloc[j, randCols_1[i]] -= unit
#             # augmentedDf.iloc[j, randCols_2[i]] = c * augmentedDf.iloc[j, randCols_1[i]] + 0
    
#     augmentedDf = pd.concat([data, augmentedDf], ignore_index= True)
#     return augmentedDf

def corrAugmentation(data, nrows, nvalues, unit):
    
    # Creates empty dataframe to store augmented data
    augmentedDf = pd.DataFrame()
    
    # Randomly selects rows from data and appends to augmentedDf
    for i in range(nrows):
        augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
        
    # Drops labels column in augmentedDf
    augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
    
    
    corrMatrix = data.corr()
    print(corrMatrix)
    
    sortedMatrix = pd.DataFrame()
    
    for i in range(1,corrMatrix.shape[0]):
        for j in range(0, i):
            lst = []
            lst.append(corrMatrix.iloc[i, j])
            lst.append(i)
            lst.append(j)
            temp = pd.DataFrame(lst)
            temp = temp.transpose()
            sortedMatrix = pd.concat([sortedMatrix, temp], axis=0, ignore_index=True)
    
    sortedMatrix = sortedMatrix.sort_values(0, ascending=False, ignore_index=True)
    
    print(sortedMatrix)
    
    cols1 = []
    cols2 = []
    corr = []
    
    for i in range(nvalues):
        cols1.append(sortedMatrix.iloc[i, 2])
        cols2.append(sortedMatrix.iloc[i, 1])
        corr.append(sortedMatrix.iloc[i, 0])
        
    
    for i in range(augmentedDf.shape[0]):
        for j in range(len(cols1)):
            # if random.randint(0, 1) == 1:
            #     augmentedDf.iloc[i, int(cols1[j])] += unit
            # else:
            #     augmentedDf.iloc[i, int(cols1[j])] -= unit
                
            c = float(corr[j]) * float(augmentedDf.iloc[i, int(cols1[j])])
            
            augmentedDf.iloc[i, 0] = c

    augmentedDf = pd.concat([data, augmentedDf], axis=0, ignore_index=True)    
    
    
    return augmentedDf
    
    
test = corrAugmentation(df, 10, 3, unit= 1)
