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



df_1 = pd.read_table("Gaussian_Distribution.txt", delimiter=" ", header=None)

dfdrop = df_1.drop(columns = df_1.shape[1] - 1)

X = dfdrop
Y = df_1[df_1.shape[1] - 1]


X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=0.2,random_state=0)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
ann.fit(X_train,Y_train,batch_size=32,epochs = 100)

predictied_values = ann.predict(X_test)

print(predictied_values)

