# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:59:12 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generateRawData import generateRawData
from fixingRandomness import applyAugmentationMethod
from superFunction import logReg
from columnNumberTesting import runClassifier

# data = generateRawData(500, 1000, .5, 'gaussian')


numCols = []
for i in range(1, 15):
    numCols.append(i)

accM = []

augAcc = []

plt.cla()

for i in range(len(numCols)):
    data = generateRawData(500, numCols[i], .5, 'uniform')
    accM.append(runClassifier(data, 'svm').iloc[0,3])
    
    feature_cols = []
    for i in range(data.shape[1]-1):
        feature_cols.append(i)
        
    pmUnit = [0, .5, 1, 5, 10]
    pmAcc = []
    
    for j in range(len(pmUnit)):

        aug = applyAugmentationMethod(df=data, method='pmOne', nrows=500, nvalues=3, unit=0.1)
    
        log = logReg(dataset=aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0]-1)
    
        acc = runClassifier(df=log, classifier='SVM')
        
        pmAcc.append(acc.iloc[0, 3])
        
    plt.plot(pmUnit, pmAcc, label=j)
    
    
    
# plt.plot(numCols, accM, label='raw')
# plt.plot(numCols, augAcc, label="augmented")
plt.legend()




