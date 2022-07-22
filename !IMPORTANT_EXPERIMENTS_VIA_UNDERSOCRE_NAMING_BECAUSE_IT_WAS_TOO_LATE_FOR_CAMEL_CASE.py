# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:10:37 2022

@author: jeffr
"""

import pandas as pd
import random

from generateRawData import generateRawData

original = generateRawData(180, 150, .05, 'uniform')

randomRow = original.iloc[[random.randint(0, original.shape[0]-1)]]
randomRow2 = original.iloc[[random.randint(0, original.shape[0]-1)]]

repeatsPresent = pd.concat([original, randomRow, randomRow2], axis=0, ignore_index=True)

from sklearn.svm import SVC
import sklearn.metrics as skm

###############################################################################
# Hyp: a
X_train = original.iloc[:150, :original.shape[1]-1]
y_train = original.iloc[:150, original.shape[1]-1]

X_test = original.iloc[150:, :original.shape[1]-1]
y_test = original.iloc[150:, original.shape[1]-1]

svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# fit the model with data
# svm.fit(X_train,y_train)
svm.fit(X_train, y_train)
predicted_values = svm.predict(X_test)

print('Hyp a:', skm.f1_score(y_test, predicted_values))


# Hyp: a + e
X_train = repeatsPresent.iloc[:160, :original.shape[1]-1]
y_train = repeatsPresent.iloc[:160, original.shape[1]-1]

X_test = repeatsPresent.iloc[160:, :original.shape[1]-1]
y_test = repeatsPresent.iloc[160:, original.shape[1]-1]

svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# fit the model with data
# svm.fit(X_train,y_train)
svm.fit(X_train, y_train)
predicted_values = svm.predict(X_test)

print('Hyp a + e:', skm.f1_score(y_test, predicted_values))

###############################################################################

original = generateRawData(160, 150, .05, 'uniform')

randomRow = original.iloc[[random.randint(0, original.shape[0]-1)]]
randomRow2 = original.iloc[[random.randint(0, original.shape[0]-1)]]
randomRow3 = original.iloc[[random.randint(0, original.shape[0]-1)]]
randomRow4 = original.iloc[[random.randint(0, original.shape[0]-1)]]

repeatsPresent = pd.concat([original, randomRow, randomRow2, randomRow3, randomRow4], axis=0, ignore_index=True)


# Hyp: a
X_train = original.iloc[:130, :original.shape[1]-1]
y_train = original.iloc[:130, original.shape[1]-1]

X_test = original.iloc[130:, :original.shape[1]-1]
y_test = original.iloc[130:, original.shape[1]-1]

svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# fit the model with data
# svm.fit(X_train,y_train)
svm.fit(X_train, y_train)
predicted_values = svm.predict(X_test)

print('Hyp a:', skm.f1_score(y_test, predicted_values))


# Hyp: a + e
X_train = repeatsPresent.iloc[:16, :original.shape[1]-1]
y_train = repeatsPresent.iloc[:16, original.shape[1]-1]

X_test = repeatsPresent.iloc[16:, :original.shape[1]-1]
y_test = repeatsPresent.iloc[16:, original.shape[1]-1]

svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# fit the model with data
# svm.fit(X_train,y_train)
svm.fit(X_train, y_train)
predicted_values = svm.predict(X_test)

print('Hyp a + e:', skm.f1_score(y_test, predicted_values))

###############################################################################

original = generateRawData(160, 150, .05, 'uniform')

randomRow = original.iloc[[random.randint(0, original.shape[0]-1)]]
randomRow2 = original.iloc[[random.randint(0, original.shape[0]-1)]]
randomRow3 = original.iloc[[random.randint(0, original.shape[0]-1)]]
randomRow4 = original.iloc[[random.randint(0, original.shape[0]-1)]]

repeatsPresent = pd.concat([randomRow, randomRow2, randomRow3, randomRow4], axis=0, ignore_index=True)


# Hyp: a
X_train = original.iloc[:130, :original.shape[1]-1]
X_train = pd.concat([X_train, repeatsPresent.iloc[:, :repeatsPresent.shape[1]-1]], axis=0, ignore_index=True)
y_train = original.iloc[:130, original.shape[1]-1]
y_train = pd.concat([y_train, repeatsPresent.iloc[:, repeatsPresent.shape[1]-1]])

X_test = original.iloc[130:, :original.shape[1]-1]
y_test = original.iloc[130:, original.shape[1]-1]

svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# fit the model with data
# svm.fit(X_train,y_train)
svm.fit(X_train, y_train)
predicted_values = svm.predict(X_test)

print('Hyp a:', skm.f1_score(y_test, predicted_values))

###############################################################################