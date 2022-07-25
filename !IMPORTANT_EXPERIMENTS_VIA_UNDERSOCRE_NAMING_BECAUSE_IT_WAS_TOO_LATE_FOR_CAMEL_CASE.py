# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:10:37 2022

@author: jeffr
"""

import pandas as pd
# import random
import numpy as np
from copy import copy

from generateRawData import generateRawData
from sklearn.svm import SVC
import sklearn.metrics as skm

## Generating 2 uniformly-distributed clusters and indexing some of them for repeatition

# original = generateRawData(180, 150, .05, 'uniform')
np.random.seed(0)
original = generateRawData(16, 2, .05, 'uniform')
spot0 = copy(original.iloc[0,:])
spot12 = copy(original.iloc[12,:])
original.iloc[0,:] = spot12
original.iloc[12,:] = spot0

# randomRow = original.iloc[[random.randint(0, original.shape[0]-1)]]
# randomRow2 = original.iloc[[random.randint(0, original.shape[0]-1)]]
rr_indx = [2,5,7,10]
repeats = original.iloc[rr_indx]

# repeatsPresent = pd.concat([original, randomRow, randomRow2], axis=0, ignore_index=True)


###############################################################################
## We'll start by establising a baseline

# Splitting data: I manually made sure these had points from the different clusters
X_train = original.iloc[:13, :original.shape[1]-1]
y_train = original.iloc[:13, original.shape[1]-1]

X_test = original.iloc[13:, :original.shape[1]-1]
y_test = original.iloc[13:, original.shape[1]-1]

svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)
svm.fit(X_train, y_train)
predicted_values = svm.predict(X_test)

print('Baseline F1-score: ', skm.f1_score(y_test, predicted_values))

del X_train, y_train, X_test, y_test, svm, predicted_values

###############################################################################
## We'll continue by looking at testing on original and testing on repeats

# Split: Original trains, repeats tests
X_train1 = original.iloc[:, :original.shape[1]-1]
y_train1 = original.iloc[:, original.shape[1]-1]

X_test1 = repeats.iloc[:, :repeats.shape[1]-1]
y_test1 = repeats.iloc[:, repeats.shape[1]-1]

svm1 = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# fit the model with data
# svm.fit(X_train,y_train)
svm1.fit(X_train1, y_train1)
predicted_values1 = svm1.predict(X_test1)

print('Original trains, repeats tests: ', skm.f1_score(y_test1, predicted_values1))

del X_train1, y_train1, X_test1, y_test1, svm1, predicted_values1

###############################################################################
## Now we'll do half of the testing on repeats and have on new data

# Split: Tests on half unseen and half repeats
X_train = pd.concat([original.iloc[:14, :original.shape[1]-1],repeats.iloc[[1,2],:repeats.shape[1]-1]], axis=0, ignore_index=True)
y_train = pd.concat([original.iloc[:14, original.shape[1]-1],repeats.iloc[[1,2],repeats.shape[1]-1]], axis=0, ignore_index=True)

X_test = pd.concat([original.iloc[14:, :original.shape[1]-1],repeats.iloc[[0,3],:repeats.shape[1]-1]], axis=0, ignore_index=True)
y_test = pd.concat([original.iloc[14:, original.shape[1]-1],repeats.iloc[[0,3],repeats.shape[1]-1]], axis=0, ignore_index=True)


svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# fit the model with data
# svm.fit(X_train,y_train)
svm.fit(X_train, y_train)
predicted_values = svm.predict(X_test)

print('Tests on half unseen and half repeats: ', skm.f1_score(y_test, predicted_values))

del X_train, y_train, X_test, y_test, svm, predicted_values

###############################################################################
## Lastly, we'll compare those with testing on all new data

# Split: Tests on all unseen
X_train = pd.concat([original.iloc[:12, :original.shape[1]-1],repeats.iloc[:, :repeats.shape[1]-1]], axis=0, ignore_index=True)
y_train = pd.concat([original.iloc[:12, original.shape[1]-1],repeats.iloc[:, repeats.shape[1]-1]], axis=0, ignore_index=True)

X_test = original.iloc[12:, :original.shape[1]-1]
y_test = original.iloc[12:, original.shape[1]-1]

svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# fit the model with data
# svm.fit(X_train,y_train)
svm.fit(X_train, y_train)
predicted_values = svm.predict(X_test)

print('Tests on all unseen: ', skm.f1_score(y_test, predicted_values))

del X_train, y_train, X_test, y_test, svm


# ###############################################################################

# X_train = original.iloc[:150, :original.shape[1]-1]
# y_train = original.iloc[:150, original.shape[1]-1]

# X_test = original.iloc[150:, :original.shape[1]-1]
# y_test = original.iloc[150:, original.shape[1]-1]


# # Hyp a:
# X_train = original.iloc[:150, :original.shape[1]-1]
# y_train = original.iloc[:150, original.shape[1]-1]

# X_test = original.iloc[150:, :original.shape[1]-1]
# y_test = original.iloc[150:, original.shape[1]-1]

# svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# # fit the model with data
# # svm.fit(X_train,y_train)
# svm.fit(X_train, y_train)
# predicted_values = svm.predict(X_test)

# print('Hyp a:', skm.f1_score(y_test, predicted_values))


# # Hyp: a + e
# X_train = repeatsPresent.iloc[:160, :original.shape[1]-1]
# y_train = repeatsPresent.iloc[:160, original.shape[1]-1]

# X_test = repeatsPresent.iloc[160:, :original.shape[1]-1]
# y_test = repeatsPresent.iloc[160:, original.shape[1]-1]

# svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# # fit the model with data
# # svm.fit(X_train,y_train)
# svm.fit(X_train, y_train)
# predicted_values = svm.predict(X_test)

# print('Hyp a + e:', skm.f1_score(y_test, predicted_values))

# ###############################################################################


# original = generateRawData(160, 150, .05, 'uniform')

# randomRow = original.iloc[[random.randint(0, original.shape[0]-1)]]
# randomRow2 = original.iloc[[random.randint(0, original.shape[0]-1)]]
# randomRow3 = original.iloc[[random.randint(0, original.shape[0]-1)]]
# randomRow4 = original.iloc[[random.randint(0, original.shape[0]-1)]]

# repeatsPresent = pd.concat([original, randomRow, randomRow2, randomRow3, randomRow4], axis=0, ignore_index=True)


# # Hyp: a
# X_train = original.iloc[:130, :original.shape[1]-1]
# y_train = original.iloc[:130, original.shape[1]-1]

# X_test = original.iloc[130:, :original.shape[1]-1]
# y_test = original.iloc[130:, original.shape[1]-1]

# svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# # fit the model with data
# # svm.fit(X_train,y_train)
# svm.fit(X_train, y_train)
# predicted_values = svm.predict(X_test)

# print('Hyp a:', skm.f1_score(y_test, predicted_values))


# # Hyp: a + e
# X_train = repeatsPresent.iloc[:16, :original.shape[1]-1]
# y_train = repeatsPresent.iloc[:16, original.shape[1]-1]

# X_test = repeatsPresent.iloc[16:, :original.shape[1]-1]
# y_test = repeatsPresent.iloc[16:, original.shape[1]-1]

# svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# # fit the model with data
# # svm.fit(X_train,y_train)
# svm.fit(X_train, y_train)
# predicted_values = svm.predict(X_test)

# print('Hyp a + e:', skm.f1_score(y_test, predicted_values))

# ###############################################################################

# original = generateRawData(160, 150, .05, 'uniform')

# randomRow = original.iloc[[random.randint(0, original.shape[0]-1)]]
# randomRow2 = original.iloc[[random.randint(0, original.shape[0]-1)]]
# randomRow3 = original.iloc[[random.randint(0, original.shape[0]-1)]]
# randomRow4 = original.iloc[[random.randint(0, original.shape[0]-1)]]

# repeatsPresent = pd.concat([randomRow, randomRow2, randomRow3, randomRow4], axis=0, ignore_index=True)


# # Hyp: a
# X_train = original.iloc[:130, :original.shape[1]-1]
# X_train = pd.concat([X_train, repeatsPresent.iloc[:, :repeatsPresent.shape[1]-1]], axis=0, ignore_index=True)
# y_train = original.iloc[:130, original.shape[1]-1]
# y_train = pd.concat([y_train, repeatsPresent.iloc[:, repeatsPresent.shape[1]-1]])

# X_test = original.iloc[130:, :original.shape[1]-1]
# y_test = original.iloc[130:, original.shape[1]-1]

# svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)

# # fit the model with data
# # svm.fit(X_train,y_train)
# svm.fit(X_train, y_train)
# predicted_values = svm.predict(X_test)

# print('Hyp a:', skm.f1_score(y_test, predicted_values))

###############################################################################