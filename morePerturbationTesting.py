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


# data = generateRawData(500,150 , -5, 'gaussian')

plt.scatter(data[0], data[1], c=data[150])
plt.show()

feature_cols = []
for i in range(data.shape[1]-1):
    feature_cols.append(i)


for i in range(1):
    aug = applyAugmentationMethod(data, 'pmOne', 500, 30, unit=10)
    log = logReg(dataset=aug, feature_cols=feature_cols, target=150, split=500)
    acc = runClassifier(log, 'SVM'), 'f1'
    
    print(acc)
    
    plt.scatter(aug[0], aug[1], c=aug[150])
    print("HELLO")