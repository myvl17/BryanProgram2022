# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 14:12:08 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generateRawData import generateRawData
from superFunction import runClassifier

'''
numRows = []
for i in range(10, 101, 2):
    numRows.append(i)

        
    
for j in range(len(numRows)):
    
    distance = [0, .25, .5, .75, 1, 2]
    accMeasure = []
    for i in range(len(distance)): 
        data = generateRawData(numRows[j], 100, distance[i], 'uniform')
        
        accMeasure.append(runClassifier(data, 'kNN').iloc[0,3])
        
    plt.plot(distance, accMeasure, marker='o')
    plt.yticks(ticks=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    plt.show()
'''
    
data = pd.read_csv('TESTING/archive (1)/diabetes.csv')

for i in range(data.shape[0]):
    if data.iloc[i, data.shape[1]-1] == 'No diabetes':
        data.iloc[i, data.shape[1]-1] = 0
    else:
        data.iloc[i, data.shape[1]-1] = 1
        
data['diabetes'] = data['diabetes'].astype('int64')

data = data.drop(['chol_hdl_ratio', 'gender', 'bmi', 'waist_hip_ratio'], axis=1)

numCols = [0,1,2,3,4,5,6,7,8,9,10,11]
data.columns = numCols

accM = []

# df = data.drop(data.columns[numCols[11]:data.shape[1]-1], axis=1)

# for i in range(3, data.shape[1]-2):
    # df = data.drop(data.columns[numCols[i]:data.shape[1]-1], axis=1)
    # acc = runClassifier(df, 'SVM').iloc[0,3]
    
#     print("THIS LINE IS HERE ", str(df.shape))
    
#     accM.append(acc)


# df = data.drop(data.columns[numCols[2]:data.shape[1]-1], axis=1)

for i in range(1, 12):
    df = data.iloc[:, :i]
    df = pd.concat([df, data[data.shape[1]-1]], axis=1, ignore_index=True)
    acc = runClassifier(df, 'SVM').iloc[0,3]
    accM.append(acc)
    
plt.plot([1,2,3,4,5,6,7,8,9,10,11], accM, marker='o')