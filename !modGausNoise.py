# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:09:38 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generateRawData import generateRawData
from columnNumberTesting import runClassifier
from betterApplyAugmentationMethod import betterApplyAugmentationMethods
from updatedSuperFunction import logReg
from modifiedGausNoise import modifiedGausNoise


# distances = np.arange(0, 2.05, .05)
# distAcc = []

# for i in range(len(distances)):
#     data = generateRawData(500, 450, distances[i], 'gaussian')
#     distAcc.append(runClassifier(data, 'SVM', 'f1'))
    
    
# fig = plt.subplots(figsize=(20,5))
# plt.plot(distances, distAcc, marker='o')
# plt.xticks(distances)
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# NROWS = 150
# NCOLS = 100

# data = generateRawData(NROWS, NCOLS, .1, 'gaussian')
# raw = runClassifier(data, 'SVM', 'f1')

# ITERATIONS = 25

# for j in range(1):
#     nValues = np.arange(0, data.shape[1]-1, 1)
#     augAcc = [0] * (len(nValues)-1)
#     augAcc.insert(0, raw)
    
#     counter = 0
    
#     for i in range(ITERATIONS):
#         for n in range(1, len(nValues)):
#             counter += 1
#             print(str(counter / (ITERATIONS * (len(nValues)-1))*100)[:4] + '%')
            
#             # aug = betterApplyAugmentationMethods(data, 'randSwap', 100, nvalues=nValues[n])
#             aug = modifiedGausNoise(data, 100, nvalues=nValues[n])
            
#             log = logReg(aug, split=data.shape[0]-1)
            
#             acc = runClassifier(log, 'SVM', 'f1')
#             augAcc[n] += acc
    
#     augAcc = np.asarray(augAcc)
#     augAcc[1:] /= ITERATIONS
#     augAcc *= 100
    
#     plt.plot(nValues, augAcc, marker='o', color='black')

# plt.xlabel('# of values swapped')
# plt.ylabel('Accuracy %')
# plt.grid(True)
# plt.yticks(np.arange(int(str(raw*100)[0] + '0'), 105, 5))
# plt.tight_layout()

# plt.show()

# fig = plt.subplots(figsize=(40,10))
# plt.plot(nValues, augAcc, marker='o', color='black')
# plt.title('modGausNoise: # of values swapped vs. Accuracy', fontsize='xx-large')
# plt.xlabel('# of values swapped', fontsize='xx-large')
# plt.ylabel('Accuracy %', fontsize='xx-large')
# plt.grid(True)
# plt.xticks(nValues, fontsize='xx-large')
# plt.yticks(np.arange(int(str(raw*100)[0] + '0'), 105, 5), fontsize='xx-large')
# plt.tight_layout()

# plt.show()


NROWS = 150
NCOLS = 100

data = generateRawData(NROWS, NCOLS, .1, 'gaussian')
raw = runClassifier(data, 'SVM', 'f1')

ITERATIONS = 25

for j in range(1):
    nRows = np.arange(0, 1050, 50)
    augAcc = [0] * (len(nRows)-1)
    augAcc.insert(0, raw)
    
    counter = 0
    
    for i in range(ITERATIONS):
        for n in range(1, len(nRows)):
            counter += 1
            print(str(counter / (ITERATIONS * (len(nRows)-1))*100)[:4] + '%')
            
            # aug = betterApplyAugmentationMethods(data, 'randSwap', nrows=nRows[n])
            aug = modifiedGausNoise(data, nrows=nRows[n], nvalues=1)
            
            log = logReg(aug, split=data.shape[0]-1)
            
            acc = runClassifier(log, 'SVM', 'f1')
            augAcc[n] += acc
    
    augAcc = np.asarray(augAcc)
    augAcc[1:] /= ITERATIONS
    augAcc *= 100
    
    plt.plot(nRows, augAcc, marker='o', color='black')

plt.xlabel('# of rows')
plt.ylabel('Accuracy %')
plt.grid(True)
plt.yticks(np.arange(int(str(raw*100)[0] + '0'), 105, 5))
plt.tight_layout()
plt.show()


fig = plt.subplots(figsize=(20,10))
plt.plot(nRows, augAcc, marker='o', color='black')
plt.title('randSwap: number of rows vs. Accuracy', fontsize='xx-large')
plt.xlabel('# of rows', fontsize='xx-large')
plt.ylabel('Accuracy %', fontsize='xx-large')
plt.grid(True)
plt.xticks(nRows)
plt.yticks(np.arange(int(str(raw*100)[0] + '0'), 105, 5), fontsize='xx-large')
plt.tight_layout()

plt.show()