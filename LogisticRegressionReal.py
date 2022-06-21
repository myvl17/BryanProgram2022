# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:12:19 2022

@author: cdiet
"""
# Import the libraries needed
import pandas as pd
import numpy as np

"""
LogReg runs original data with labels and take augmented data without
labels, runs them through a logistic regression, and outputs predicted
labels. 

LogReg Inputs:
    dataset = Saved txt or csv file with original data, augmented data,
    and labels 
    feature_cols = All columns that are not the labels, listed as an array
    target = the labels or status column, a string, the name of that column
    in the dataframe
    split = The number of rows in the txt/csv that are original and not 
    augmented data
    save = The file where you wish to save the labels
    
LogReg Outputs:
    Outputs a text file that contains all of the predicted labels for the 
    augmented data.
"""
from sklearn.model_selection import train_test_split
import sklearn.metrics 

def LogReg(dataset, feature_cols, target, split):
        
    # Feature variables
    X = dataset[feature_cols]
    
    # Target variable
    y = dataset[target]
    
    # Split both x and y into training and testing sets
    
    # Splitting the sets
    # Test_size indicates the ratio of testing to training data ie 0.25:0.75
    # Random_state indicates to select data randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = split, shuffle = False,  stratify = None) 
    
    
    # import the class
    from sklearn.linear_model import LogisticRegression
    
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(max_iter = 10000)
    
    # fit the model with data
    logreg.fit(X_train,y_train)
    
    # create the prediction
    y_pred= logreg.predict(X_test)
    
    
    # Appends predicted labels to NAN
    for i in range(split, dataset.shape[0]):
        dataset.loc[i, target] = y_pred[split-i]
    
    return dataset


## EXAMPLE

# feature_cols = []
# for i in range(0, 149, 1):
#     feature_cols.append(i)
    
# complete_df = LogReg(dataset = augmented,
#                feature_cols = feature_cols, target = 150, split = 500)
