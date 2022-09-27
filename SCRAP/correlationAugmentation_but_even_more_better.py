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

# synthetic dataset

x = [.25, .75, .4, .8, .9, 1, 1.1, .95, .925, .85]
y = [.3, .2, .75, .6, .75, .5, .75, .9, 1, 1.1]
z = [.17, .48, .32, .58, .63, 1.26, .98, .87, 1.19, 1.0]

# labels

labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# create dataframe

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


# method

def corrAugmentation(data, nrows, nvalues, unit):
    
    # Creates empty dataframe to store augmented data
    augmentedDf = pd.DataFrame()
    
    # Randomly selects rows from data and appends to augmentedDf
    for i in range(nrows):
        augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
        
    # Drops labels column in augmentedDf
    augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
    
    # Correlation matrix of our data without labels
    corrMatrix = data.iloc[:, :data.shape[1]-1].corr()
    # print(corrMatrix)
    
    # Create empty dataframe for correalation values.
    sortedMatrix = pd.DataFrame()
    
    # Create dataframe with correlation values and between two columns and column numbers.
    for i in range(1,corrMatrix.shape[0]):
        for j in range(0, i):
            lst = []
            lst.append(corrMatrix.iloc[i, j])
            lst.append(i)
            lst.append(j)
            temp = pd.DataFrame(lst)
            temp = temp.transpose()
            sortedMatrix = pd.concat([sortedMatrix, temp], axis=0, ignore_index=True)
    
    # Ordering from highest to lowest correlation
    
    sortedMatrix = sortedMatrix.sort_values(0, ascending=False, ignore_index=True)
    
    #print(sortedMatrix)
    
    cols1 = []
    cols2 = []
    corr = []
    
    # Create list of columns and correlations. 
    
    for i in range(nvalues):
        cols1.append(sortedMatrix.iloc[i, 2])
        cols2.append(sortedMatrix.iloc[i, 1])
        corr.append(sortedMatrix.iloc[i, 0])
        
    # Looping through each augmented row in the dataframe
    
    for i in range(augmentedDf.shape[0]):
        for j in range(len(cols1)):
            
            print(cols1[j], ',', cols2[j])
            
            # Choosing random value (0,1) if 1 add, if 0 subtract
            
            if random.randint(0, 1) == 1:
                
            # Choosing the row and column value and adding or subtracting
                  augmentedDf.iloc[i, int(cols1[j])] += unit
            else:
                  augmentedDf.iloc[i, int(cols1[j])] -= unit
                
            temp = pd.concat([data, augmentedDf.iloc[[i]]], axis=0, ignore_index=True)
            tempCorr = temp.corr().iloc[int(cols1[j]), int(cols2[j])]
            print(temp.corr())
            print(tempCorr)
            print(corr[j])
            counter = 0
            while ((tempCorr < corr[j] - .01)  | (tempCorr > corr[j] + .01)):
                temp.iloc[i, int(cols2[j])] -= .0001
                
                tempCorr = temp.corr().iloc[int(cols1[j]), int(cols2[j])]
                print(tempCorr)
                #print("sup")
                counter+= 1
                
            
    augmentedDf = pd.concat([data, augmentedDf], axis=0, ignore_index=True)    
    
    
    return augmentedDf
    
    
test = corrAugmentation(df, 1, 1, unit= 1)
test2 = test.iloc[:, :test.shape[1]-1].corr()
# print(test2)



"""
          0         1         2
0  1.000000  0.417451  0.843807
1  0.417451  1.000000  0.543281
2  0.843807  0.543281  1.000000
          0         1         2
0  1.000000  0.115807  0.252127
1  0.115807  1.000000  0.481483
2  0.252127  0.481483  1.000000
"""