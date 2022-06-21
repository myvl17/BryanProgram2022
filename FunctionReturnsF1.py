# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:57:39 2022

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
import random

def OkayFunction(data, accuracy=None):
    
    txt = "txt"
    if txt in data:
        df = pd.read_table(data, delimiter = " ", header = None) 
    
    else:
        df = data
        
    dfdrop = df.drop(columns = df.shape[1] - 1)
    
    # results_df = pd.DataFrame(columns = ["Accuracy", "Mean Absolute Error", "Rooted Mean Square Error", "F1 Score"])
    
    X = dfdrop
    y = df[df.shape[1] - 1]

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
                 X, y, test_size = 0.2, random_state=0)
     
    random.seed(1)
    knn = KNeighborsClassifier(n_neighbors=7)
     
    knn.fit(X_train, y_train)
     
    # Predict on dataset which model has not seen before
    predicted_values = knn.predict(X_test)
        
    #Accuracy
    f1_accuracy = skm.f1_score(y_test, predicted_values)
        
        
    return f1_accuracy
    
    
