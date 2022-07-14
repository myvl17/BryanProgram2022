# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 19:15:49 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from generateRawData import generateRawData

from superFunction import logReg
from columnNumberTesting import runClassifier
from modifiedGausNoise import modifiedGausNoise



NROWS = 100
NCOLS = 75

data = generateRawData(NROWS, NCOLS, .15, 'gaussian')
raw = runClassifier(data, 'SVM', 'f1')
print(raw)

ITERATIONS = 10

for j in range(1):
    nCols = np.arange(0, NCOLS+1, 1)
    augAcc = [0] * (len(nCols)-1)
    augAcc.insert(0, raw)
    
    counter = 0
    
    for i in range(ITERATIONS):
        for n in range(1, len(nCols)):
            counter += 1
            print(str(counter / (ITERATIONS * (len(nCols)-1))*100)[:4] + '%')
            
            aug = modifiedGausNoise(data, 100, nCols[n])
            
            log = logReg(aug, feature_cols = np.arange(0, aug.shape[1]-1), target=aug.shape[1]-1, split=data.shape[0]-1)
            
            acc = runClassifier(log, 'SVM', 'f1')
            augAcc[n] += acc
    
    augAcc = np.asarray(augAcc)
    augAcc[1:] /= ITERATIONS
    augAcc *= 100
    
    plt.plot(nCols, augAcc, marker='o', color='black')

plt.xlabel('# Columns Swapped')
plt.ylabel('Accuracy %')
plt.grid(True)
plt.yticks(np.arange(int(str(raw*100)[0] + '0'), 105, 5))
plt.tight_layout()
plt.show()


# NROWS = 100
# NCOLS = 75

# data = generateRawData(NROWS, NCOLS, .15, 'gaussian')
# raw = runClassifier(data, 'SVM', 'f1')
# print(raw)

# ITERATIONS = 100

# for j in range(1):
#     nRows = np.arange(50, 1500, 50)
#     augAcc = [0] * (len(nRows)-1)
#     augAcc.insert(0, raw)
    
#     counter = 0
    
#     for i in range(ITERATIONS):
#         for n in range(1, len(nRows)):
#             counter += 1
#             print(str(counter / (ITERATIONS * (len(nRows)-1))*100)[:4] + '%')
            
#             aug = modifiedGausNoise(data, nRows[n], 3)
            
#             log = logReg(aug, feature_cols = np.arange(0, aug.shape[1]-1), target=aug.shape[1]-1, split=data.shape[0]-1)
            
#             acc = runClassifier(log, 'SVM', 'f1')
#             augAcc[n] += acc
    
#     augAcc = np.asarray(augAcc)
#     augAcc[1:] /= ITERATIONS
#     augAcc *= 100
    
#     plt.plot(nRows, augAcc, marker='o', color='black')

# plt.title('# of New Observations vs. Accuracy')
# plt.xlabel('# Rows Augmented')
# plt.ylabel('Accuracy %')
# plt.grid(True)
# plt.yticks(np.arange(int(str(raw*100)[0] + '0'), 105, 5))
# plt.tight_layout()
# plt.show()