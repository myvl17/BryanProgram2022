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

raw = generateRawData(500, 150, .25, 'gaussian')
rawAcc = runClassifier(raw, 'SVM', 'f1')


fig, ax = plt.subplots(4, 3, sharey=True, sharex=False, figsize=(20, 10))


distances = np.arange(0, 1.05, 0.05)

exp1Acc = []
exp2Acc = []
exp3Acc = []


for i in range(len(distances)):
    data = generateRawData(500, 150, distances[i], 'gaussian')
    
    
    # EXPERIMENT ONE
    exp1Acc.append(runClassifier(data,'SVM', 'f1'))
    
    

    # EXPERIMENT TWO
    x = data.iloc[:, :2]
    y = data.iloc[:, data.shape[1]-1]
    xy = pd.concat([x, y], axis=1)
    
    exp2Acc.append(runClassifier(xy, 'SVM', 'f1'))

    
    # EXPERIMENT THREE
    df = StandardScaler().fit_transform(data)
    pcaExp = PCA(n_components=2)
    PCs = pcaExp.fit_transform(data)
    
    PCdf = pd.DataFrame(PCs)
    
    newDf = pd.concat([PCdf, data.iloc[:, data.shape[1]-1]], axis=1)
    
    exp3Acc.append(runClassifier(newDf, 'SVM', 'f1'))
    
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

exp1AccNoise = [rawAcc]
exp2AccNoise = [rawAcc]
exp3AccNoise = [rawAcc]

ITERATIONS = 2

for i in range(0, len(noise)-1):
    exp1AccNoise.append(0)
    exp2AccNoise.append(0)
    exp3AccNoise.append(0)

for j in range(ITERATIONS):
    percent = int(j / ITERATIONS * 100)
    print("COMPLETION: ", percent, "%")
    
    for i in range(1, len(noise)):
        
        # EXPERIMENT ONE
        data = generateRawData(500, 150, 0.25, 'gaussian')
        
        feature_cols = np.arange(0, data.shape[1]-1, 1)
        
        aug = applyAugmentationMethod(data, 'gausNoise', 100, 0, noise=noise[i])
        log = logReg(aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0]-1)
        acc = runClassifier(log, 'SVM', 'f1')
        exp1AccNoise[i] = exp1AccNoise[i] + acc
        
        
        # EXPERIMENT TWO
        data = generateRawData(500, 150, 0.25, 'gaussian')
        x = data.iloc[:, :2]
        y = data.iloc[:, data.shape[1]-1]
        xy = pd.concat([x, y], axis=1, ignore_index=True)
        
        feature_cols = np.arange(0, xy.shape[1]-1, 1)
        
        aug = applyAugmentationMethod(xy, 'gausNoise', 100, 0, noise=noise[i])
        log = logReg(aug, feature_cols=feature_cols, target=xy.shape[1]-1, split=xy.shape[0]-1)
        acc = runClassifier(log, 'SVM', 'f1')
        
        exp2AccNoise[i] = exp2AccNoise[i] + acc
        
        
        # EXPERIMENT THREE
        data = generateRawData(500, 150, 0.25, 'gaussian')
        df = StandardScaler().fit_transform(data)
        pcaExp = PCA(n_components=2)
        PCs = pcaExp.fit_transform(data)
        
        PCdf = pd.DataFrame(PCs)
        
        newDf = pd.concat([PCdf, data.iloc[:, data.shape[1]-1]], axis=1, ignore_index=True)
        
        aug = applyAugmentationMethod(newDf, 'gausNoise', 100, 0, noise=noise[i])
        log = logReg(aug, feature_cols=[0,1], target=newDf.shape[1]-1, split=newDf.shape[0]-1)
        acc = runClassifier(log, 'SVM', 'f1')
        
        exp3AccNoise[i] = exp3AccNoise[i] + acc

print("COMPLETION:  100 %")
npExp1AccNoise = np.asarray(exp1AccNoise)
npExp1AccNoise[1:] /= ITERATIONS
npExp2AccNoise = np.asarray(exp2AccNoise)
npExp2AccNoise[1:] /= ITERATIONS
npExp3AccNoise = np.asarray(exp3AccNoise)
npExp3AccNoise[1:] /= ITERATIONS

ax[1,0].set_ylabel('gausNoise')
ax[1,0].plot(noise, npExp1AccNoise, marker='o')
ax[1,0].grid(True)
ax[1,0].set_xlabel('Noise')
ax[1,1].plot(noise, npExp2AccNoise, marker='o')
ax[1,1].grid(True)
ax[1,1].set_xlabel('Noise')
ax[1,2].plot(noise, npExp3AccNoise, marker='o')
ax[1,2].grid(True)
ax[1,2].set_xlabel('Noise')



units = np.arange(0, 2.05, .05)

exp1AccPM = [rawAcc]
exp2AccPM = [rawAcc]
exp3AccPM = [rawAcc]

ITERATIONS = 2

for i in range(0, len(units)-1):
    exp1AccPM.append(0)
    exp2AccPM.append(0)
    exp3AccPM.append(0)

for j in range(ITERATIONS):
    percent = int(j / ITERATIONS * 100)
    print("COMPLETION: ", percent, "%")
    
    for i in range(1, len(units)):
        data = generateRawData(500, 150, 0.25, 'gaussian')
        
        # EXPERIMENT ONE
        feature_cols = np.arange(0, data.shape[1]-1, 1)
        
        aug = applyAugmentationMethod(data, 'pmOne', 100, 30, unit=units[i])
        log = logReg(aug, feature_cols=feature_cols, target=data.shape[1]-1, split=data.shape[0]-1)
        acc = runClassifier(log, 'SVM', 'f1')
        
        exp1AccPM[i] = exp1AccPM[i] + acc
        
        
        # EXPERIMENT TWO
        data = generateRawData(500, 150, 0.25, 'gaussian')
        x = data.iloc[:, :2]
        y = data.iloc[:, data.shape[1]-1]
        xy = pd.concat([x, y], axis=1, ignore_index=True)
        
        feature_cols = np.arange(0, xy.shape[1]-1, 1)
        
        aug = applyAugmentationMethod(xy, 'pmOne', 100, 1, unit=units[i])
        log = logReg(aug, feature_cols=feature_cols, target=xy.shape[1]-1, split=xy.shape[0]-1)
        acc = runClassifier(log, 'SVM', 'f1')
        
        exp2AccPM[i] = exp2AccPM[i] + acc
        
        
        # EXPERIMENT THREE
        data = generateRawData(500, 150, 0.25, 'gaussian')
        df = StandardScaler().fit_transform(data)
        pcaExp = PCA(n_components=2)
        PCs = pcaExp.fit_transform(data)
        
        PCdf = pd.DataFrame(PCs)
        
        newDf = pd.concat([PCdf, data.iloc[:, data.shape[1]-1]], axis=1, ignore_index=True)
        
        aug = applyAugmentationMethod(newDf, 'pmOne', 100, 1, unit=units[i])
        log = logReg(aug, feature_cols=[0,1], target=newDf.shape[1]-1, split=newDf.shape[0]-1)
        acc = runClassifier(log, 'SVM', 'f1')
        
        exp3AccPM[i] = exp3AccPM[i] + acc
    
print('COMPLETION:  100 %')
npExp1AccPM = np.asarray(exp1AccPM)
npExp1AccPM[1:] /= ITERATIONS
npExp2AccPM = np.asarray(exp2AccPM)
npExp2AccPM[1:] /= ITERATIONS   
npExp3AccPM = np.asarray(exp3AccPM)
npExp3AccPM[1:] /= ITERATIONS   

ax[2,0].set_ylabel('pmOne')
ax[2,0].plot(units, npExp1AccPM, marker='o')
ax[2,0].grid(True)
ax[2,0].set_xlabel('Units (30 values changes)')
ax[2,1].plot(units, npExp2AccPM, marker='o')
ax[2,1].grid(True)
ax[2,1].set_xlabel('Units (1 value changed)')
ax[2,2].plot(units, npExp3AccPM, marker='o')
ax[2,2].grid(True)
ax[2,2].set_xlabel('Units (1 value changed)')


plt.tight_layout()
plt.show()

    
    
    