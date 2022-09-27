# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:55:27 2022

@author: cdiet
"""

# Import the libraries needed
import pandas as pd
import numpy as np
import random

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

def SVC(dataset, feature_cols, target, split):
        
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
    from sklearn.svm import SVC
    
    # instantiate the model (using the default parameters)
    random.seed(1)
    svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)
    
    # fit the model with data
    # svm.fit(X_train,y_train)
    svm.fit(dataset.iloc[:5, :2], dataset.iloc[:5, 2])
    tmp2 = svm.predict(dataset.iloc[5:, :2])
    
    
    # create the prediction
    # y_pred= svm.predict(X_test)

    # Appends predicted labels to NAN
    for i in range(split, dataset.shape[0]):
        dataset.loc[i, target] = tmp2[i-split]
    
    return dataset

