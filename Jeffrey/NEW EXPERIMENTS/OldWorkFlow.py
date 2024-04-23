# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 09:21:20 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from generateRawData import generateRawData
from betterApplyAugmentationMethod import betterApplyAugmentationMethods
from updatedSuperFunction import superFunction
from updatedSuperFunction import logReg
from updatedSuperFunction import runClassifier

data = generateRawData(500, 250, .15, 'gaussian')


accuracy = []
nvalues = np.arange(10, 260, 10)

accuracy.append(runClassifier(data, 'SVM', 'f1') * 100)

for i in nvalues:
    print(i)
    augmented = betterApplyAugmentationMethods(data, 'randswap', nrows=200, nvalues=i)
    labels = logReg(augmented, data.shape[0]-1)
    f1score = runClassifier(labels, 'SVM', 'f1')
    accuracy.append(f1score * 100)
    
nvalues = np.insert(nvalues, 0, 0)


fig = plt.subplots(figsize=(30,10))
plt.plot(nvalues, accuracy, marker='o', color='black', label="Old Workflow")
plt.title('Old Workflow: randSwap: # of values swapped vs. Accuracy', fontsize='xx-large')
plt.xlabel('# of values swapped', fontsize='xx-large')
plt.ylabel('Accuracy %', fontsize='xx-large')
plt.grid(True)
plt.xticks(nvalues, fontsize='xx-large')
plt.yticks(np.arange(int(str(65*100)[0] + '0'), 105, 5))
plt.legend()
plt.tight_layout()

plt.show()

###############################################################################

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sklearn.metrics as skm

accuracyNew = []
nvaluesNew = np.arange(10, 260, 10)

X_train, X_test, y_train, y_test = train_test_split(
             data.iloc[:, :data.shape[1]-1], data.iloc[:, data.shape[1]-1], test_size = 0.2, random_state=42)

# Concat to create complete training and testing set
training = pd.concat([X_train, y_train], axis=1)
testing = pd.concat([X_test, y_test], axis=1)


accuracyNew.append(runClassifier(training, 'SVM', 'f1')*100)


for i in nvaluesNew:
    print(i)
    augmented = betterApplyAugmentationMethods(training, 'randSwap', 200, nvalues=i)
    logRegression = logReg(augmented, training.shape[0])
    
    # New training set with augmented data
    X_train = logRegression.iloc[:, :logRegression.shape[1]-1]
    y_train = logRegression.iloc[:, logRegression.shape[1]-1]
    
    
    dfdrop = logRegression.drop(columns = logRegression.shape[1] - 1)
    
    X = dfdrop
    y = logRegression[logRegression.shape[1] - 1]
    
    # random.seed(1)
    svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 1000000, random_state = 0)
    
    # fit the model with data
    # svm.fit(X_train,y_train)
    svm.fit(X_train, y_train)
    predicted_values = svm.predict(X_test)
    
    f1_accuracy = skm.f1_score(y_test, predicted_values)
    accuracyNew.append(f1_accuracy*100)

nvaluesNew = np.insert(nvaluesNew, 0, 0)

fig = plt.subplots(figsize=(30,10))
plt.plot(nvaluesNew, accuracyNew, marker='o', color='red', label="New Workflow")
plt.title('New Workflow: randSwap: # of values swapped vs. Accuracy', fontsize='xx-large')
plt.xlabel('# of values swapped', fontsize='xx-large')
plt.ylabel('Accuracy %', fontsize='xx-large')
plt.grid(True)
plt.xticks(nvaluesNew, fontsize='xx-large')
plt.yticks(np.arange(int(str(65*100)[0] + '0'), 105, 5))
plt.legend()
plt.tight_layout()

plt.show()

