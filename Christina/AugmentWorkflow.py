# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:16:47 2022

@author: cdiet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from SyntheticData import UniformSythetic

UniformSythetic(500, 150, 2)

from UniformAugmentation import RandUnit

# Run the function
print(RandUnit('synthetic_data_with_labels.txt', 500, 0.1))   
       

from LogisticRegressionReal import LogReg

feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)
print(LogReg(dataset = 'augmented_original_label.txt',
              feature_cols = feature_cols, target = 'status', split = 500,
              save = 'augmented_data_labels.txt'))

# Take the labels from the original data, append the predicted labels
# Add that column to original and augmented data

data = pd.read_table('augmented_original.txt', delimiter = " ", header = None)
original_label = pd.read_table('synthetic_data_labels.txt', delimiter = " ", header = None)

augmented_label = pd.read_table('augmented_data_labels.txt', delimiter = " ", header = None)

labels =  pd.concat([original_label, augmented_label])

# data['status'] = labels

# # Output to txt
# np.savetxt('all_data.txt', data)

#KNN
 
# Loading data
dfknn = pd.read_table('augmented_original.txt', delimiter = " ", header = None)

# Create feature and target arrays
X = dfknn
y = labels
 
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
 
knn = KNeighborsClassifier(n_neighbors=7)
 
knn.fit(X_train, y_train)

# Predict on dataset which model has not seen before
predicted_values = (knn.predict(X_test))

# ACCURACY

import sklearn.metrics as skm


accuracy = skm.accuracy_score(y_test, predicted_values)
print(accuracy)

ame_accuracy = skm.mean_absolute_error(y_test, predicted_values)
print(ame_accuracy)

rmse_accuracy = skm.mean_squared_error(y_test, predicted_values)
print(rmse_accuracy)




