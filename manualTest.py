# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:43:46 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = [.25, .75, .4, .8, .9, 1, 1.1, .95, .925, .85]
y = [.3, .2, .75, .6, .75, .5, .75, .9, 1, 1.1]

labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

plt.scatter(x, y, c=labels)

df = pd.DataFrame({0:x, 1:y, 2:labels})



from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

'''
0
1
4

5
7
9
'''
    
training = [0,1,4,5,7,9]
test = [2,3,6,8]

X_train = df.iloc[training, :2]
y_train = df.iloc[training, 2]

X_test = df.iloc[test, :2]
y_test = df.iloc[test, 2]

knn = KNeighborsClassifier(n_neighbors=2)
 
knn.fit(X_train, y_train)
 
# Predict on dataset which model has not seen before
predicted_values = knn.predict(X_test)

from sklearn.metrics import f1_score

print(f1_score(y_test, predicted_values))


from fixingRandomness import applyAugmentationMethod

augmented = applyAugmentationMethod(df=df, method='pmOne', nrows=4, nvalues=1, unit=0.1)

from superFunction import logReg

labels = logReg(augmented, [0,1], 2, 10)


test = [2,3,6,8,9,10,11,12,13]

X_train = labels.iloc[training, :2]
y_train = labels.iloc[training, 2]

X_test = labels.iloc[test, :2]
y_test = labels.iloc[test, 2]

knn = KNeighborsClassifier(n_neighbors=2)
 
knn.fit(X_train, y_train)
 
# Predict on dataset which model has not seen before
predicted_values = knn.predict(X_test)

print(f1_score(y_test, predicted_values))
    
    
plt.scatter(labels[0], labels[1], c=labels[2])
plt.show()

plt.scatter(X_test[0], X_test[1], c=y_test-predicted_values)
    
    
    
    
    
    