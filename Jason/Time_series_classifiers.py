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



df = pd.read_csv("epilepsy_data copy.csv")

df_1 = df[0: 500]

df_2 = df_1.drop(columns = ['Unnamed: 0'])
              
df_2.loc[df_2['y'] != 1, 'y'] = 0


df_2.to_csv('binary_epilepsy_2.1.csv')


numCols = []
acc = []
for i in range(2,178):
    numCols.append(i)

for i in range(len(numCols)):
    X = df_2.iloc[:, 0:numCols[i]]
    Y = df_2['y']
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,Y,test_size=.2,random_state=42, shuffle= True)
    
    
    
knn = KNeighborsClassifier(metric='dtw')
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
    
    
clf = TimeSeriesForest(random_state=43)
clf.fit(X_train, y_train)
    
acc.append(clf.score(X_test, y_test))

plt.plot(numCols[:i+1], acc)
plt.show()





