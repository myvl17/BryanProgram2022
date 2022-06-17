# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:05:02 2022

@author: cdiet
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import random
import sklearn.metrics as skm

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def OkayFunction(data, accuracy):
    df = pd.read_table(data, delimiter = " ", header = None) 
    dfdrop = df.drop(columns = df.shape[1] - 1)
    
    X = dfdrop
    y = df[df.shape[1] - 1]
    
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
                 X, y, test_size = 0.2, random_state=42)
     
    knn = KNeighborsClassifier(n_neighbors=7)
     
    knn.fit(X_train, y_train)
     
    # Predict on dataset which model has not seen before
    predicted_values = knn.predict(X_test)
        
    #Accuracy
    if (accuracy == "og"): 
        acc = skm.metrics.accuracy_score(y_test, predicted_values)
        return acc
    
    elif (accuracy == "mae"):
        mae_accuracy = skm.mean_absolute_error(y_test, predicted_values)
        return mae_accuracy
    
    elif (accuracy == "rmae"):
        rmae_accuracy = skm.mean_squared_error(y_test, predicted_values,
                                                    squared=False)
        return rmae_accuracy
    
    elif(accuracy == "f1"):
        f1_accuracy = skm.f1_score(y_test, predicted_values)
        return f1_accuracy
    
    
print(OkayFunction('synthetic_data_with_labels.txt', "f1"))