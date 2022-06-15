# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:12:19 2022

@author: cdiet
"""
# Import the libraries needed
import pandas as pd
import numpy as np

"""
LogReg Inputs
dataset: saved txt or csv file with augmented data and labels 
name: what to call the dataset once it is read in
feature_cols: All columns that are not the labels
target: the labels or status column, a string, the name of that column
in the dataframe
split: the number of the row in the dataframe where it goes from 
original to augmented data
save: what to save the output labels as


Output: Will be an 
"""

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

## EXAMPLE

feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)
    
print(LogReg(dataset = 'augmented_original_label.txt', name = "data",
             feature_cols = feature_cols, target = 'status', split = 500,
             save = 'augmented_data_labels.txt'))

