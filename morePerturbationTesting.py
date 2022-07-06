# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 08:17:34 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fixingRandomness import applyAugmentationMethod
from generateRawData import generateRawData
from superFunction import logReg
from superFunction import runClassifier


# data = generateRawData(500, 150 , -1.1, 'uniform')

print(runClassifier(data, 'SVM', 'f1'))

# plt.scatter(data[0], data[1], c=data[data.shape[1]-1])
# plt.show()
# plt.close()

feature_cols = []
for i in range(data.shape[1]-1):
    feature_cols.append(i)



pmOneUnit = [0, 0.1, 0.2, 0.5, 0.75]
pmAccuracy = []

for i in range(len(pmOneUnit)):
    aug = applyAugmentationMethod(data, 'pmOne', 500, 30, unit=pmOneUnit[i])
    log = logReg(dataset=aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0])
    acc = runClassifier(log, 'SVM', 'f1')
    
    
    # print(acc.iloc[0, 3])
    pmAccuracy.append(acc.iloc[0, 3])
    
    # plt.scatter(aug[0], aug[1], c=aug[aug.shape[1]-1])
    # plt.show()
    
# plt.cla()
plt.plot(pmOneUnit, pmAccuracy)
plt.show()


# gausNoise = [0, 0.05, .25, .5, .75, 1]
# gausNoiseAcc = []

# for i in range(len(gausNoise)):
#     aug = applyAugmentationMethod(data, 'gausNoise', 500, 30, noise=gausNoise[i])
#     log = logReg(dataset=aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0])
#     acc = runClassifier(log, 'SVM', 'f1')
    
    
#     # print(acc.iloc[0, 3])
#     gausNoiseAcc.append(acc.iloc[0, 3])
    
#     # plt.scatter(aug[0], aug[1], c=aug[aug.shape[1]-1])
#     # plt.show()
    
# plt.plot(gausNoise, gausNoiseAcc)
# plt.show()


# randSwap = [0, 1, 5, 25, 50, 75, 100, 150]
# randSwapAcc = []

# for i in range(len(randSwap)):
#     aug = applyAugmentationMethod(data, 'randSwap', 500, randSwap[i])
#     log = logReg(dataset=aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0])
#     acc = runClassifier(log, 'SVM', 'f1')
    
    
#     # print(acc.iloc[0, 3])
#     randSwapAcc.append(acc.iloc[0, 3])
    
#     # plt.scatter(aug[0], aug[1], c=aug[aug.shape[1]-1])
#     # plt.show()
    
# plt.plot(randSwap, randSwapAcc)
# plt.show()