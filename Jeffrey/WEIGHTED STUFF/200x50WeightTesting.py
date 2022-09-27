# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:32:48 2022

@author: jeffr
"""

'''
NOTE TO SELF BECAUSE I'LL PROBABLY FORGET:

I SPLIT THE DATA FIRST TO KEEP THE TRAINING SET CONSISTENT BEFORE AND AFTER AUGMENTATION
ONLY DIFFERENCE IN TRAINING SET BEFORE AND AFTER IS ADDITIONAL ROWS OF AUGMENTED POINTS
DOING SO WILL RESULT IN BETTER COMPARISON WITH ORIGINAL DATA VS AUGMENTED ???


I DONT KNOW HOW WEIGHTED SAMPLES WORK


'''


from generateRawData import generateRawData
from updatedSuperFunction import runClassifier
from updatedSuperFunction import logReg
from betterApplyAugmentationMethod import betterApplyAugmentationMethods
import pandas as pd


data = generateRawData(200, 135, .1, 'gaussian')


from sklearn.model_selection import train_test_split

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             data.iloc[:, :data.shape[1]-1], data.iloc[:, data.shape[1]-1], test_size = 0.2, random_state=42)

# Concat to create complete training and testing set
training = pd.concat([X_train, y_train], axis=1)
testing = pd.concat([X_test, y_test], axis=1)


initialAcc = runClassifier(training, 'SVM', 'f1')
augmented = betterApplyAugmentationMethods(training, 'pmone', 50, unit=2)
logRegression = logReg(augmented, training.shape[0])

# New training set with augmented data
X_train = logRegression.iloc[:, :logRegression.shape[1]-1]
y_train = logRegression.iloc[:, logRegression.shape[1]-1]

weight1 = [2]*(X_train.shape[0]-50)
weight2 = [1]*50

weight = weight1 + weight2


from sklearn.svm import SVC
import sklearn.metrics as skm

dfdrop = logRegression.drop(columns = logRegression.shape[1] - 1)

X = dfdrop
y = logRegression[logRegression.shape[1] - 1]

# random.seed(1)
svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 1000000, random_state = 0)

# fit the model with data
# svm.fit(X_train,y_train)
svm.fit(X_train, y_train, sample_weight=weight)
predicted_values = svm.predict(X_test)

f1_accuracy = skm.f1_score(y_test, predicted_values)
print(f1_accuracy)





