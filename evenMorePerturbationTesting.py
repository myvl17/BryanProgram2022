# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:59:12 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generateRawData import generateRawData
from superFunction import applyAugmentationMethod
from superFunction import logReg
from columnNumberTesting import runClassifier

data = generateRawData(500, 7, .25, 'gaussian')

accRaw = runClassifier(data, 'svm').iloc[0,3]


# numCols = [1,2,3,4,5,6,7]

# accM = []

# augAcc = []

# plt.cla()

# for i in range(len(numCols)):
#     data = generateRawData(500, numCols[i], .5, 'uniform')
#     accM.append(runClassifier(data, 'svm').iloc[0,3])
    
#     feature_cols = []
#     for i in range(data.shape[1]-1):
#         feature_cols.append(i)
        
#     pmUnit = [0, .5, 1, 5, 10]
#     pmAcc = []
    
#     label = str(numCols[i]) + " cols"
    
#     for j in range(len(pmUnit)):

#         aug = applyAugmentationMethod(df=data, method='pmOne', nrows=500, nvalues=3, unit=pmUnit[j])
    
#         log = logReg(dataset=aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0]-1)
    
#         acc = runClassifier(df=log, classifier='SVM')
        
#         pmAcc.append(acc.iloc[0, 3])
        
#     plt.plot(pmUnit, pmAcc, label=label)
    
    
# plt.legend()


feature_cols = []
for i in range(data.shape[1]-1):
    feature_cols.append(i)
    
pmUnit = [0, .1, 0.25, 0.5, 0.75, 1]
pmAcc = [accRaw]

for j in range(1,len(pmUnit)):

    aug = applyAugmentationMethod(data, method='pmOne', nrows=500, nvalues=3, unit=pmUnit[j])

    log = logReg(dataset=aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0]-1)

    acc = runClassifier(df=log, classifier='SVM')
    
    pmAcc.append(acc.iloc[0, 3])
    
plt.plot(pmUnit, pmAcc)
plt.title("pmOne Perturbation Amount Accuracy")
plt.show()


gausNoise = [0, .05, .25, .5, .75, 1]
gausAcc = [accRaw]

for j in range(1,len(gausNoise)):

    aug = applyAugmentationMethod(data, method='gausNoise', nrows=500, nvalues=3, noise=gausNoise[j])

    log = logReg(dataset=aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0]-1)

    acc = runClassifier(df=log, classifier='SVM')
    
    gausAcc.append(acc.iloc[0, 3])
    
plt.plot(gausNoise, gausAcc)
plt.title("gausNoise Perturbation Amount Accuracy")
plt.show()


randAmount = [0,1,2,3,4,5,6,7]
randAcc = [accRaw]

for j in range(1, len(randAmount)):

    aug = applyAugmentationMethod(data, method='randSwap', nrows=500, nvalues=randAmount[j])

    log = logReg(dataset=aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0]-1)

    acc = runClassifier(df=log, classifier='SVM')
    
    randAcc.append(acc.iloc[0, 3])
    
plt.plot(randAmount, randAcc)
plt.title("randSwap Perturbation Amount Accuracy")
plt.show()







