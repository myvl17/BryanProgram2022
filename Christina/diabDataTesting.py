# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:37:29 2022

@author: cdiet
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fakeSuper import applyAugmentationMethod
from betterLogRegpy import betterLogReg
from columnNumberTesting import runClassifier
from evenBetterRandSwap import betterRandSwap
from betterGausNoise import betterGausNoise
from modifiedGausNoise import modifiedGausNoise
from modPMOne import modPMOne

## VARIABLES
rows = 150
cols = 100
ITERATIONS = 100
counter = 0

pmUnit = np.arange(0, 10 , 0.25)
randUnit = np.arange(0, cols, 1)
gausNoise = np.arange(0, 5.25, 0.25)
modGausNoise = np.arange(0, cols, 1)
rowIter = np.arange(0, 1050, 50)
modPMOneUnit = np.arange(0, 5.25, 0.25)

df100 = pd.read_table("diabData.txt", header = None, delimiter = " ")


# Find the feature cols
cols1 = []
for i in range(0, cols - 1, 1):
    cols1.append(i)
    
## Run accuracy on original dataset
acc1 = runClassifier(df100, 'SVM', 'f1')
# accFin.append(acc1)
            

## Instantiate the list for the accuracies
# augAcc = [0] * (len(randUnit) - 1)
# augAcc.insert(0, acc1)

# augAcc2 = [0] * (len(pmUnit) - 1)
# augAcc2.insert(0, acc1)

# augAcc3 = [0] * (len(gausNoise) - 1)
# augAcc3.insert(0, acc1)

# augAcc4 = [0] * (len(modGausNoise) - 1)
# augAcc4.insert(0, acc1)

# augAcc5 = [0] * (len(modPMOneUnit) - 1)
# augAcc5.insert(0, acc1)

augAcc = [0] * (len(rowIter) - 1)
augAcc.insert(0, acc1)

augAcc2 = [0] * (len(rowIter) - 1)
augAcc2.insert(0, acc1)

augAcc4 = [0] * (len(rowIter) - 1)
augAcc4.insert(0, acc1)

augAcc3 = [0] * (len(rowIter) - 1)
augAcc3.insert(0, acc1)

augAcc5 = [0] * (len(rowIter) - 1)
augAcc5.insert(0, acc1)


# ## Experiment for finding the optimal percent of rows to augment
# # finAcc = []  

# # for data in data:
# #     dftest = pd.concat([dflast.iloc[:data, :data], diab2[:data]], axis = 1, ignore_index = True)
# #     rows = np.arange(0, data, 5)
# #     acc1 = runClassifier(dftest, 'SVM', 'f1')
# #     augAcc3 = [0] * (len(rows) - 1)
# #     augAcc3.insert(0, acc1)
    
# #     cols1 = []
# #     for i in range(0, data - 1, 1):
# #         cols1.append(i)
        
    
# #     for j in range(ITERATIONS):
# #         for i in range(1, len(rows)):
# #             counter += 1
        
# #             print(str(counter / (ITERATIONS * (len(rows)-1))*100)[:4] + '%')
# #             x = applyAugmentationMethod(dftest, 'pmOne', rows[i], 20, unit = 1)
# #             log = betterLogReg(dataset = x, feature_cols = cols1, target = data, split = data)
# #             acc = runClassifier(log, 'SVM', 'f1')
# #             augAcc3[i] += acc
            
        
# #     augAcc3 = np.asarray(augAcc3)
# #     augAcc3[1:] /= ITERATIONS
# #     augAcc3 *= 100
    
# #     print(max(augAcc3))
# #     print(data)
    
# #     augAcc3 = augAcc3.tolist()
# #     finAcc.append((rows[augAcc3.index(max(augAcc3))]/ data )* 100)

# #     fig, ax = plt.subplots()
# #     ax.plot(rows, augAcc3, marker = 'o', linewidth = 3.0, color = 'white')
# #     ax.set_title('Accuracy vs Rows Augmented')
# #     ax.set_xlabel('Unit')
# #     ax.set_ylabel('Accuracy')
# #     ax.set_facecolor('magenta')
# #     plt.show()
  
## Experiment for accuracy and pmOne perturbation  
# for j in range(ITERATIONS):
#     for i in range(1, len(rowIter)):
#         counter += 1
    
#         print(str(counter / (ITERATIONS * (len(rowIter)-1))*100)[:4] + '%')
#         x = applyAugmentationMethod(df100, 'pmOne', rowIter[i], cols, unit = 0)
#         log = betterLogReg(dataset = x, feature_cols = cols1, target = cols, split = rows)
#         acc = runClassifier(log, 'SVM', 'f1')
#         augAcc2[i] += acc
        
    
# augAcc2 = np.asarray(augAcc2)
# augAcc2[1:] /= ITERATIONS
# augAcc2 *= 100

# for q in range(ITERATIONS):
#     for r in range(1, len(rowIter)):
#         counter += 1
    
#         print(str(counter / (ITERATIONS * (len(rowIter)-1))*100)[:4] + '%')
#         x = modPMOne(df100, rowIter[r], cols, 3)
#         log = betterLogReg(dataset = x, feature_cols = cols1, target = cols, split = rows)
#         acc = runClassifier(log, 'SVM', 'f1')
#         augAcc5[r] += acc
        
    
# augAcc5 = np.asarray(augAcc5)
# augAcc5[1:] /= ITERATIONS
# augAcc5 *= 100

# # ## Experiment for accuracy and randUnit perturbation
# for k in range(ITERATIONS):
#     for l in range(1, len(rowIter)):
#         counter += 1
#         print(str(counter / (ITERATIONS * (len(rowIter)-1))*100)[:4] + '%')
#         # x = applyAugmentationMethod(dftest, 'randSwap', 50, value)
#         x = betterRandSwap(df100, rowIter[l], 95)
#         log = betterLogReg(dataset = x, feature_cols = cols1, target = cols, split = rows)
#         acc = runClassifier(log, 'SVM', 'f1')
#         augAcc[l] += acc
    
# augAcc = np.asarray(augAcc)
# augAcc[1:] /= ITERATIONS
# augAcc *= 100

# # ## Experiment for accuracy and randUnit perturbation
# for m in range(ITERATIONS):
#     for n in range(1, len(rowIter)):
#         counter += 1
#         print(str(counter / (ITERATIONS * (len(rowIter)-1))*100)[:4] + '%')
#         # x = applyAugmentationMethod(dftest, 'randSwap', 50, value)
#         x = betterGausNoise(df100, rowIter[n], cols, noise = 0.25)
#         log = betterLogReg(dataset = x, feature_cols = cols1, target = cols, split = rows)
#         acc = runClassifier(log, 'SVM', 'f1')
#         augAcc3[n] += acc
    
# augAcc3 = np.asarray(augAcc3)
# augAcc3[1:] /= ITERATIONS
# augAcc3 *= 100

## Experiment for accuracy and randUnit perturbation
for o in range(ITERATIONS):
    for p in range(1, len(rowIter)):
        counter += 1
        print(str(counter / (ITERATIONS * (len(rowIter)-1))*100)[:4] + '%')
        # x = applyAugmentationMethod(dftest, 'randSwap', 50, value)
        x = modifiedGausNoise(df100, rowIter[p], 90)
        log = betterLogReg(dataset = x, feature_cols = cols1, target = cols, split = rows)
        acc = runClassifier(log, 'SVM', 'f1')
        augAcc4[p] += acc
    
augAcc4 = np.asarray(augAcc4)
augAcc4[1:] /= ITERATIONS
augAcc4 *= 100

# ##Graphs of experiments vs accuracy        
# fig, ax = plt.subplots()
# ax.plot(rowIter, augAcc2, marker = 'o', linewidth = 3.0, color = 'blue')
# ax.set_title('Accuracy vs pmOne Row Accuracy')
# ax.set_xlabel('Unit')
# ax.set_ylabel('Accuracy')
# ax.set_facecolor('white')
# plt.show()

# plt.plot(rowIter, augAcc, marker = 'o', linewidth = 3.0, color = 'blue')
# plt.title('Accuracy vs randUnit Row Accuracy')
# plt.xlabel('Rows')
# plt.ylabel('Accuracy')

# plt.show()

# plt.plot(rowIter, augAcc3, marker = 'o', linewidth = 3.0, color = 'blue')
# plt.title('Accuracy vs gausNoise Row Accuracy')  
# plt.xlabel('Rows')
# plt.ylabel('Accuracy')
# plt.show()

plt.plot(rowIter, augAcc4, marker = 'o', linewidth = 3.0, color = 'blue')
plt.title('Accuracy vs modGausNoise Row Accuracy')
plt.xlabel('Rows')
plt.ylabel('Accuracy')
plt.show()

# plt.plot(rowIter, augAcc5, marker = 'o', linewidth = 3.0, color = 'blue')
# plt.title('Accuracy vs modPMOne Row Accuracy')
# plt.xlabel('Rows')
# plt.ylabel('Accuracy')
# plt.show()

