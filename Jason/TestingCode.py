#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:57:48 2022

@author: jasonwhite
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import random
import sklearn.metrics as skm


# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from sklearn.preprocessing import StandardScaler



df = pd.read_table("combined_data.txt", delimiter=" ", header=None)



dfdrop = df.drop(columns = df.shape[1] - 1)


X = dfdrop
Y = df[df.shape[1] - 1]


#Splitting dataset into training and testing dataset
X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=7,random_state=42, shuffle= False)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

ann.fit(X_train,Y_train,batch_size=32,epochs = 100)

predicted_values = ann.predict(X_test)
print(predicted_values>.5)
predicted_labels = predicted_values > .5
final_predicted_labels  = predicted_labels * 1
print(final_predicted_labels)


X = dfdrop
Y = df[df.shape[1] - 1]
 
 # Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
              X, Y, test_size = 5, random_state=42, shuffle = False)
  
knn = KNeighborsClassifier(n_neighbors=3)
  
knn.fit(X_train, y_train)
  
 # Predict on dataset which model has not seen before
predicted_values = knn.predict(X_test)
print(predicted_values)
