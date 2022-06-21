# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:48:14 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

"""
randSwap Inputs

file: file name, NOTE: DELIMETER IS SET TO SPACES AND THERE IS NO HEADER
perturbations: number of new rows to generate
ncols: number of columns to substitute with random value
"""

def randSwap(file, nrows, nvalues):
    # Reads .txt data frame file
    df = pd.read_table(file, delimiter=" ", header=None)
    
    # Creates empty data frame
    augmented_df = pd.DataFrame()
    
    for k in range(0, nrows, 1):
               
        # Selects random row index
        random_row = random.randint(0, df.shape[0]-1)
        
        # Adds new row from pre-existing random row
        augmented_df = pd.concat([augmented_df, df.iloc[[random_row]]], ignore_index=True)
        
        
        # Actual Data Augmentation Method:
        # Grabs random row from original data set and appends to new data frame
        # Selects random column from new row and takes random value from same column in original data set
        # Appends random value from original data frame and appends to new row column in new data frame
        for i in range(nvalues):
            
            # Selects random column index
            random_col = random.randint(0, df.shape[1]-2)
            
            # Selects random value from original data frame in the same column
            rand_value = df.iloc[random.randint(0, df.shape[0]-1)][random_col] # BREAKS THINGS
            
            # Appends rand_value to new column
            augmented_df.iloc[-1][random_col] = rand_value
            
    # Removes label column
    augmented_df.drop(df.columns[-1], axis=1, inplace=True)
    
    finished_df = pd.concat([df, augmented_df], ignore_index=True)
    
    return finished_df

augmented = randSwap("Generated Gaussian Distribution.txt", 100, 30)


"""

Graph to visualize augmented data frame versus original

"""
fix, ax = plt.subplots(1,2, sharey=True)

ax[0].hist(augmented)
ax[0].set_title("Augmented")


ax[1].hist(pd.read_table("Generated Gaussian Distribution.txt", delimiter=" ", header=None))
ax[1].set_title("Original")
plt.show()



plt.scatter(pd.read_table("Generated Gaussian Distribution.txt", delimiter=" ", header=None)[0], pd.read_table("Generated Gaussian Distribution.txt", delimiter=" ", header=None)[1], alpha=0.6, c="b", label="original")
plt.scatter(augmented[0], augmented[1], alpha=0.2, c="r", label="augmented")
plt.title("Augmented Random Gaussian Distribution")
plt.legend()

plt.show()


"""

Log regression

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

    
feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)

complete_df = LogReg(dataset = augmented,
               feature_cols = feature_cols, target = 150, split = 500)


from sklearn.neighbors import KNeighborsClassifier

def knnClassifier(dataframe, feature_cols, target):
    X = dataframe[feature_cols]
    y = dataframe[target]
     
     
    knn = KNeighborsClassifier(n_neighbors=7)
     
    knn.fit(X, y)
     
    # Predict on dataset which model has not seen before
    predictions = knn.predict(X)
    
    return predictions

predictions = knnClassifier(complete_df, feature_cols, 150)


def accuracy(file, predictions):
    # Grabs last column containing labels
    return sklearn.metrics.accuracy_score(file[150], predictions)
    
print(accuracy(complete_df, predictions))
    