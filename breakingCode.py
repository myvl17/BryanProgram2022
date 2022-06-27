# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:26:01 2022

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

# rawData1 = [0.25, 3.47, 0.74, 0.56, 3.20]
# rawData2 = [0.42, 3.85, 0.89, 0.17, 3.98]
# labels = [0, 1, 0, 1, 0]



# table = pd.DataFrame({'x':rawData1, 'y':rawData2, 'labels':labels})
# table = table.reset_index(drop = True)
# np.savetxt('rawData.txt', table)

from generateRawData import generateRawData
from fixingRandomness import applyAugmentationMethod
from SVC import SVC
from superFunction import logReg
from FunctionReturnsF1 import OkayFunction
from superFunction import superFunction

np.savetxt('smallGausDist.txt', generateRawData(500, 2, -10, 'gaussian'))

# print(SVC((applyAugmentationMethod('rawData.txt', 'pmOne', 5, 1, unit = 0.1)), 
#         [0, 1], 2, 5))


# print(superFunction(file = 'rawData.txt', method = 'gausNoise', nrows = 5, nvalues=1,
#                     feature_cols = [0, 1], target = 2, split = 4, classifier = 
#                     'kNN', noise = 0.05))

augment = applyAugmentationMethod(df = 'smallGausDist.txt', method = "gausNoise", nrows = 500, nvalues = 2, noise = 0.05)

df = logReg(augment, feature_cols = [0, 1], target = 2, split = 500)
    
df2 = pd.read_table('smallGausDist.txt', delimiter = " ", header = None) 

plt.scatter(df2[0], df2[1], c = df2[df2.shape[1] - 1])
plt.show()
plt.scatter(df[0], df[1], c = df[df.shape[1] - 1])
        
# dfdrop = df.drop(columns = df.shape[1] - 1)

# print(dfdrop)
# # results_df = pd.DataFrame(columns = ["Accuracy", "Mean Absolute Error", "Rooted Mean Square Error", "F1 Score"])

# X = dfdrop
# y = df[df.shape[1] - 1]

# print(X)
# print(y)

# # Split into training and test set
# X_train, X_test, y_train, y_test = train_test_split(
#              X, y, test_size = 0.5, random_state=0)
 
# print(X_train)
# print(X_test )
# print( y_train)
# print(y_test)

# random.seed(1)
# knn = KNeighborsClassifier(n_neighbors=1)
 
# knn.fit(X_train, y_train)
 
# # Predict on dataset which model has not seen before
# predicted_values = knn.predict(X_test)

# print(predicted_values)

# acc = skm.accuracy_score(y_test, predicted_values)

# print(acc)

    