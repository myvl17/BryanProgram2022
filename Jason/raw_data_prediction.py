#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:00:49 2022

@author: jasonwhite
"""

import pandas as pd

import sklearn.metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df_1 = pd.read_table("Gaussian_Distribution.txt", delimiter=" ", header=None)
df_2 = pd.read_table("Gaussian_noise.txt", delimiter=" ", header=None)
df_3 = pd.read_table("augmented_data_labels.txt", delimiter = " ", header = None)

df_2[150] = df_3
print(df_2)

df = pd.concat([df_1, df_2])
dfdrop = df.drop(columns = df.shape[1] - 1)



feature_cols = []
for i in range(150):
    feature_cols.append(i)
    
X = df[feature_cols]

y = df[150]

X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
 
knn = KNeighborsClassifier(n_neighbors=7)
 
knn.fit(X_train, y_train)
 
# Generates predictions
pred = knn.predict(X_test)

# Prints basic accuracy
print(sklearn.metrics.accuracy_score(pred, y_test))

