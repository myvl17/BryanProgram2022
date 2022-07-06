#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:35:27 2022

@author: jasonwhite
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import random
import sklearn.metrics as skm
from sklearn.datasets import make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sea
from pyts.classification import KNeighborsClassifier
from pyts.datasets import load_gunpoint
from pyts.classification import TimeSeriesForest



df = pd.read_csv("Binary_epilepsy.csv")



df_1 = df.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'])
              
# df_2 = df_1.loc[df_1['y'] == 1].sample(n = 4)

# print(df_2)

# df_3 = df_1[df_1['y'] != 1]

# df_4 = pd.concat([df_2, df_3])

# print(df_4)

numCols = []
acc = []
for i in range(2,178):
    numCols.append(i)

for i in range(len(numCols)):
    X = df_1.iloc[:, 0:numCols[i]]
    Y = df_1['y']
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,Y,test_size=.2,random_state=42, shuffle= True)
    
    
    
    # knn = KNeighborsClassifier(metric='dtw')
    # knn.fit(X_train, y_train)
    
    # knn.score(X_test, y_test)
    
    
    clf = TimeSeriesForest(random_state=43)
    clf.fit(X_train, y_train)
    
    acc.append(clf.score(X_test, y_test))

    plt.plot(numCols[:i+1], acc)
    plt.show()





