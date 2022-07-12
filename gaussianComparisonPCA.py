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
    x = data.iloc[:, :2]
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
    
ax[0,0].set_ylabel('Raw')
ax[0,0].plot(distances, exp1Acc, marker='o')
ax[0,0].set_xlabel('Distance')
ax[0,0].grid(True)
ax[0,0].set_title('500 x 150')
ax[0,1].plot(distances, exp2Acc, marker='o')
ax[0,1].set_xlabel('Distance')
ax[0,1].grid(True)
ax[0,1].set_title('500 x 2')
ax[0,2].plot(distances, exp3Acc, marker='o')
ax[0,2].set_xlabel('Distance')
ax[0,2].grid(True)
ax[0,2].set_title('500 x PCA')

    
noise = np.arange(0, 2.05, .05)
exp1AccNoise = [exp1Acc[0]]
exp2AccNoise = [exp2Acc[0]]
exp3AccNoise = [exp3Acc[0]]


for i in range(1, len(noise)):
    data = generateRawData(500, 150, 0.25, 'gaussian')
    
    # EXPERIMENT ONE
    feature_cols = np.arange(0, data.shape[1]-1, 1)
    
    aug = applyAugmentationMethod(data, 'gausNoise', 100, 0, noise=noise[i])
    log = logReg(aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0]-1)
    acc = runClassifier(log, 'SVM', 'f1').iloc[0,3]
    exp1AccNoise.append(acc)
    
    
    # EXPERIMENT TWO
    data = generateRawData(500, 150, 0.25, 'gaussian')
    x = data.iloc[:, :2]
    y = data.iloc[:, data.shape[1]-1]
    xy = pd.concat([x, y], axis=1, ignore_index=True)
    
    feature_cols = np.arange(0, xy.shape[1]-1, 1)
    
    aug = applyAugmentationMethod(xy, 'gausNoise', 100, 0, noise=noise[i])
    log = logReg(aug, feature_cols=feature_cols, target=xy.shape[1]-1, split=xy.shape[0]-1)
    acc = runClassifier(log, 'SVM', 'f1').iloc[0,3]
    
    exp2AccNoise.append(acc)
    
    
    # EXPERIMENT THREE
    data = generateRawData(500, 150, 0.25, 'gaussian')
    df = StandardScaler().fit_transform(data)
    pcaExp = PCA(n_components=2)
    PCs = pcaExp.fit_transform(data)
    
    PCdf = pd.DataFrame(PCs)
    
    newDf = pd.concat([PCdf, data.iloc[:, data.shape[1]-1]], axis=1, ignore_index=True)
    
    aug = applyAugmentationMethod(newDf, 'gausNoise', 100, 0, noise=noise[i])
    log = logReg(aug, feature_cols=[0,1], target=newDf.shape[1]-1, split=newDf.shape[0]-1)
    acc = runClassifier(log, 'SVM', 'f1').iloc[0,3]
    
    exp3AccNoise.append(acc)

ax[1,0].set_ylabel('gausNoise')
ax[1,0].plot(noise, exp1AccNoise, marker='o')
ax[1,0].grid(True)
ax[1,0].set_xlabel('Noise')
ax[1,1].plot(noise, exp2AccNoise, marker='o')
ax[1,1].grid(True)
ax[1,1].set_xlabel('Noise')
ax[1,2].plot(noise, exp3AccNoise, marker='o')
ax[1,2].grid(True)
ax[1,2].set_xlabel('Noise')



units = np.arange(0, 2.05, .05)
exp1AccPM = [exp1Acc[0]]
exp2AccPM = [exp2Acc[0]]
exp3AccPM = [exp3Acc[0]]

for i in range(1, len(units)):
    data = generateRawData(500, 150, 0.25, 'gaussian')
    
    # EXPERIMENT ONE
    feature_cols = np.arange(0, data.shape[1]-1, 1)
    
    aug = applyAugmentationMethod(data, 'pmOne', 100, 30, unit=units[i])
    log = logReg(aug, feature_cols=feature_cols, target=xy.shape[1]-1, split=data.shape[0]-1)
    acc = runClassifier(log, 'SVM', 'f1').iloc[0,3]
    
    exp1AccPM.append(acc)
    
    
    # EXPERIMENT TWO
    data = generateRawData(500, 150, 0.25, 'gaussian')
    x = data.iloc[:, :2]
    y = data.iloc[:, data.shape[1]-1]
    xy = pd.concat([x, y], axis=1, ignore_index=True)
    
    feature_cols = np.arange(0, xy.shape[1]-1, 1)
    
    aug = applyAugmentationMethod(xy, 'pmOne', 100, 1, unit=units[i])
    log = logReg(aug, feature_cols=feature_cols, target=xy.shape[1]-1, split=xy.shape[0]-1)
    acc = runClassifier(log, 'SVM', 'f1').iloc[0,3]
    
    exp2AccPM.append(acc)
    
    
    # EXPERIMENT THREE
    data = generateRawData(500, 150, 0.25, 'gaussian')
    df = StandardScaler().fit_transform(data)
    pcaExp = PCA(n_components=2)
    PCs = pcaExp.fit_transform(data)
    
    PCdf = pd.DataFrame(PCs)
    
    newDf = pd.concat([PCdf, data.iloc[:, data.shape[1]-1]], axis=1, ignore_index=True)
    
    aug = applyAugmentationMethod(newDf, 'pmOne', 100, 1, unit=units[i])
    log = logReg(aug, feature_cols=[0,1], target=newDf.shape[1]-1, split=newDf.shape[0]-1)
    acc = runClassifier(log, 'SVM', 'f1').iloc[0,3]
    
    exp3AccPM.append(acc)


plt.tight_layout()
plt.show()

    
    
    