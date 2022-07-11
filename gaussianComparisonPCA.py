# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:55:23 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from generateRawData import generateRawData
from columnNumberTesting import runClassifier

# from fakeSuper import runClassifier

from fixingRandomness import applyAugmentationMethod
from superFunction import logReg

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


fig, ax = plt.subplots(4, 3, sharey=True, sharex=False, figsize=(20, 10))


distances = np.arange(0, 1.05, 0.05)

exp1Acc = []
exp2Acc = []
exp3Acc = []


for i in range(len(distances)):
    data = generateRawData(500, 150, distances[i], 'gaussian')
    
    
    # EXPERIMENT ONE
    exp1Acc.append(runClassifier(data,'SVM', 'f1').iloc[0,3])
    
    

    # EXPERIMENT TWO
    x = data.iloc[:, :3]
    y = data.iloc[:, data.shape[1]-1]
    xy = pd.concat([x, y], axis=1)
    
    exp2Acc.append(runClassifier(xy, 'SVM', 'f1').iloc[0,3])

    
    # EXPERIMENT THREE
    df = StandardScaler().fit_transform(data)
    pcaExp = PCA(n_components=2)
    PCs = pcaExp.fit_transform(data)
    
    PCdf = pd.DataFrame(PCs)
    
    newDf = pd.concat([PCdf, data.iloc[:, data.shape[1]-1]], axis=1)
    
    exp3Acc.append(runClassifier(newDf, 'SVM', 'f1').iloc[0,3])
    
    
ax[0,0].plot(distances, exp1Acc, marker='o')
ax[0,0].grid(True)
ax[0,0].set_title('500 x 150')
ax[0,1].plot(distances, exp2Acc, marker='o')
ax[0,1].grid(True)
ax[0,1].set_title('500 x 2')
ax[0,2].plot(distances, exp3Acc, marker='o')
ax[0,2].grid(True)
ax[0,2].set_title('500 x PCA')

    
noise = np.arange(0, 2.05, .05)
exp1AccNoise = []
exp2AccNoise = []
exp3AccNoise = []


for i in range(len(noise)):
    data = generateRawData(500, 150, 0.25, 'gaussian')
    
    feature_cols = np.arange(0, data.shape[1]-1, 1)
    
    aug = applyAugmentationMethod(data, 'gausNoise', 100, 0, noise=noise[i])
    log = logReg(aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0]-1)
    acc = runClassifier(log, 'SVM', 'f1').iloc[0,3]
    exp1AccNoise.append(acc)
    
    
    # EXPERIMENT TWO
    x = data.iloc[:, :3]
    y = data.iloc[:, data.shape[1]-1]
    xy = pd.concat([x, y], axis=1)
    
    aug = applyAugmentationMethod(data, 'gausNoise', 100, 0, noise=noise[i])
    log = logReg(aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0]-1)
    acc = runClassifier(log, 'SVM', 'f1').iloc[0,3]
    
    exp2AccNoise.append(acc)
    
    
    # EXPERIMENT THREE
    df = StandardScaler().fit_transform(data)
    pcaExp = PCA(n_components=2)
    PCs = pcaExp.fit_transform(data)
    
    PCdf = pd.DataFrame(PCs)
    
    newDf = pd.concat([PCdf, data.iloc[:, data.shape[1]-1]], axis=1)
    
    aug = applyAugmentationMethod(newDf, 'gausNoise', 100, 0, noise=noise[i])
    log = logReg(aug, feature_cols=[0,1], target=data.shape[1]-1, split=data.shape[0]-1)
    acc = runClassifier(log, 'SVM', 'f1').iloc[0,3]
    
    exp3AccNoise.append(acc)
    
ax[1,0].plot(noise, exp1AccNoise, marker='o')
ax[1,1].plot(noise, exp2AccNoise, marker='o')
ax[1,2].plot(noise, exp3AccNoise, marker='o')
    
    
    
    
    