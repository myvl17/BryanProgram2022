# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:09:36 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generateRawData import generateRawData
from columnNumberTesting import runClassifier
from betterApplyAugmentationMethod import betterApplyAugmentationMethods
from updatedSuperFunction import logReg


# distances = np.arange(0, 2.05, .05)
# distAcc = []

# for i in range(len(distances)):
#     data = generateRawData(500, 450, distances[i], 'uniform')
#     distAcc.append(runClassifier(data, 'SVM', 'f1'))
    
    
# fig = plt.subplots(figsize=(20,5))
# plt.plot(distances, distAcc, marker='o')
# plt.xticks(distances)
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# NROWS = 150
# NCOLS = 100

# data = generateRawData(NROWS, NCOLS, .05, 'uniform')
# raw = runClassifier(data, 'SVM', 'f1')

# ITERATIONS = 25

# for j in range(1):
#     units = np.arange(0, 10.25, .25)
#     augAcc = [0] * (len(units)-1)
#     augAcc.insert(0, raw)
    
#     counter = 0
    
#     for i in range(ITERATIONS):
#         for n in range(1, len(units)):
#             counter += 1
#             print(str(counter / (ITERATIONS * (len(units)-1))*100)[:4] + '%')
            
#             aug = betterApplyAugmentationMethods(data, 'pmOne', 100, unit=units[n])
            
#             log = logReg(aug, split=data.shape[0]-1)
            
#             acc = runClassifier(log, 'SVM', 'f1')
#             augAcc[n] += acc
    
#     augAcc = np.asarray(augAcc)
#     augAcc[1:] /= ITERATIONS
#     augAcc *= 100
    
#     plt.plot(units, augAcc, marker='o', color='black')

# plt.xlabel('Units')
# plt.ylabel('Accuracy %')
# plt.grid(True)
# plt.yticks(np.arange(int(str(raw*100)[0] + '0'), 105, 5))
# plt.tight_layout()

# plt.show()

# fig = plt.subplots(figsize=(20,10))
# plt.plot(units, augAcc, marker='o', color='black')
# plt.title('pmOne Units vs. Accuracy', fontsize='xx-large')
# plt.xlabel('Units', fontsize='xx-large')
# plt.xticks(units, fontsize='large')
# plt.ylabel('Accuracy %', fontsize='xx-large')
# plt.grid(True)
# plt.yticks(np.arange(int(str(raw*100)[0] + '0'), 105, 5))
# plt.tight_layout()

# plt.show()


NROWS = 150
NCOLS = 100

data = generateRawData(NROWS, NCOLS, .05, 'uniform')
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
            
            aug = betterApplyAugmentationMethods(data, 'pmOne', nRows[n], unit = .25)
            
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
plt.xlabel('# of rows', fontsize='xx-large')
plt.title('Number of rows vs. Accuracy', fontsize='xx-large')
plt.xticks(nRows, fontsize='xx-large')
plt.ylabel('Accuracy %', fontsize='xx-large')
plt.grid(True)
plt.yticks(np.arange(int(str(raw*100)[0] + '0'), 105, 5), fontsize='xx-large')
plt.tight_layout()

plt.show()
