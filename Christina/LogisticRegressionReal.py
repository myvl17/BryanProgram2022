# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:12:19 2022

@author: cdiet
"""

import pandas as pd
import numpy as np

def LogReg(dataset, name, feature_cols, target, split, save):
    # read in the file
    txt = "txt"
    csv = "csv"
    if csv in dataset:
        name = pd.read_csv(dataset)
        
    elif txt in dataset:
        name = pd.read_table(dataset, delimiter = " ", header = None)
        name.rename(columns = {150: target}, inplace = True)
        
    # Find the ratio 
    ratio = split / len(name)

    # Feature variables
    X = name[feature_cols]
    
    # Target variable
    y = name[target]
    
    # Split both x and y into training and testing sets
    
    # Import train_test_split
    from sklearn.model_selection import train_test_split
    
    # Splitting the sets
    # Test_size indicates the ratio of testing to training data ie 0.25:0.75
    # Random_state indicates to select data randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, shuffle = False,  stratify = None) 
    
    
    # import the class
    from sklearn.linear_model import LogisticRegression
    
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(max_iter = 10000)
    
    # fit the model with data
    logreg.fit(X_train,y_train)
    
    # create the prediction
    y_pred= logreg.predict(X_test)
    
    np.savetxt(save, y_pred)

    return y_pred

