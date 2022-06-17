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


def OkayFunction(data, accuracy=None):
    df = pd.read_table(data, delimiter = " ", header = None) 
    dfdrop = df.drop(columns = df.shape[1] - 1)
    
    results_df = pd.DataFrame(columns = ["Accuracy", "Mean Absolute Error", "Rooted Mean Square Error", "F1 Score"])
    
    X = dfdrop
    y = df[df.shape[1] - 1]
    
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
                 X, y, test_size = 0.2, random_state=42)
     
    knn = KNeighborsClassifier(n_neighbors=7)
     
    knn.fit(X_train, y_train)
     
    # Predict on dataset which model has not seen before
    predicted_values = knn.predict(X_test)

    print(predicted_values)
        
    #Accuracy
    if (accuracy == "og"): 
        acc = skm.accuracy_score(y_test, predicted_values)
        
    elif (accuracy == "mae"):
        mae_accuracy = skm.mean_absolute_error(y_test, predicted_values)

    
    elif (accuracy == "rmse"):
        rmse_accuracy = skm.mean_squared_error(y_test, predicted_values,
                                                    squared=False)

    
    elif(accuracy == "f1"):
        f1_accuracy = skm.f1_score(y_test, predicted_values)

        
    else:
        acc = skm.accuracy_score(y_test, predicted_values)
        mae_accuracy = skm.mean_absolute_error(y_test, predicted_values)
        rmse_accuracy = skm.mean_squared_error(y_test, predicted_values,
                                                    squared=False)
        f1_accuracy = skm.f1_score(y_test, predicted_values)
        
        results_df.append({'Accuracy':acc, 
                           'Mean Absolute Error':mae_accuracy,
                           'Rooted Mean Square Error':rmse_accuracy,
                           'F1 Score':f1_accuracy}, ignore_index=True)
        
    return results_df
    
    
df = OkayFunction('Generated Gaussian Distribution.txt')