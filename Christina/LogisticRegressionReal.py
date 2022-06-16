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

def LogReg(dataset, feature_cols, target, split, save):
    # read in the file
    txt = "txt"
    csv = "csv"
    if csv in dataset:
        data = pd.read_csv(dataset)
        
    elif txt in dataset:
        data = pd.read_table(dataset, delimiter = " ", header = None)
        data.rename(columns = {150: target}, inplace = True)
        
    # Find the ratio so it knows what percent is test vs training
    ratio = (split / len(data))

    # Feature variables
    X = data[feature_cols]
    
    # Target variable
    y = data[target]
    
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
    
    # Save the predictions as text files for later use
    np.savetxt(save, y_pred)

## EXAMPLE

# feature_cols = []
# for i in range(0, 149, 1):
#     feature_cols.append(i)
    
# print(LogReg(dataset = 'augmented_original_label.txt', name = "data",
#              feature_cols = feature_cols, target = 'status', split = 500,
#              save = 'augmented_data_labels.txt'))

