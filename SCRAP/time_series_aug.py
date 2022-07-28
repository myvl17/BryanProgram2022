#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:59:33 2022

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

from fixingRandomness import applyAugmentationMethod
from ts_decision_tree import ts_dtree
from betterGausNoise import betterGausNoise
from betterPmOne import betterPmOne
from evenBetterRandSwap import betterRandSwap
from modifiedGausNoise import modifiedGausNoise
from modPMOne import modPMOne
feature_cols = []
for i in range(0, 178, 1):
     feature_cols.append(i)

df = pd.read_csv("binary_epilepsy_2.1.csv")

df_1 = df.drop(columns = ['Unnamed: 0'], axis=1)

df_2 = df_1.rename({'y': 178}, axis = 1)

df_2.columns = range(df_2.shape[1])

df_2.T.reset_index(drop = True).T

df_3 = df_2[0:200]

loop_list = np.arange(50, 1000, 50)

acc = [0] * len(loop_list)

ITERATIONS = 10


for j in range(ITERATIONS):
    for i in range(len(loop_list)):
    
        aug_df = modPMOne(df_3, loop_list[i], df_3.shape[1]-1, unit = 1.00)
        
        dtree_df = ts_dtree(aug_df, target=178, split=199)
        
        
        dfdrop = dtree_df.drop(columns = dtree_df.shape[1] - 1)
        
        X = dfdrop
        Y = dtree_df[dtree_df.shape[1] - 1]
         
         
        X_train, X_test, y_train, y_test = train_test_split(
            X,Y,test_size=.2,random_state=42, shuffle= True)
         
         
         
        knn = KNeighborsClassifier(metric='dtw')
        knn.fit(X_train, y_train)
         
        predicted_values = knn.predict(X_test)
        
        from sklearn.metrics import f1_score
        # acc.append(f1_score(y_test, predicted_values))
        acc[i] += f1_score(y_test, predicted_values)

acc = np.asarray(acc)
acc /= ITERATIONS



 
plt.plot(loop_list, acc, color = 'hotpink', linewidth = 6.0)
plt.ylabel('Average Accuracy')
plt.xlabel('Rows Augmented')
plt.title('ModPmOne 1.00 unit, 25 Iterations')
plt.show()



"""

acc_1 = [0] * ITERATIONS

for i in range(ITERATIONS):
    
    aug_df = betterGausNoise(df_3, 650, df_3.shape[1]-1, noise = 7.5)
    
    dtree_df = ts_dtree(aug_df, target=178, split=199)
    
    
    dfdrop = dtree_df.drop(columns = dtree_df.shape[1] - 1)
    
    X = dfdrop
    Y = dtree_df[dtree_df.shape[1] - 1]
     
     
    X_train, X_test, y_train, y_test = train_test_split(
        X,Y,test_size=.2,random_state=42, shuffle= True)
     
     
     
    knn = KNeighborsClassifier(metric='dtw')
    knn.fit(X_train, y_train)
     
    predicted_values = knn.predict(X_test)
    
    from sklearn.metrics import f1_score
    # acc.append(f1_score(y_test, predicted_values))
    acc_1[i] += f1_score(y_test, predicted_values)

original_accuracy = sum(acc_1) / ITERATIONS

"""