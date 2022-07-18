# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:45:31 2022

@author: cdiet
"""

from fakeSuper import superFunction
import matplotlib.pyplot as plt
from generateRawData import generateRawData
from fakeSuper import runClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from fakeSuper import logReg
from fakeSuper import applyAugmentationMethod


feature_cols =[]
for i in range(0, 149, 1):
    feature_cols.append(i)
    

# Empty lists for plots
Gaussian = []
uniform = []
Gaussian2 = []
uniform2 = []
Gaussian3 = []
uniform3 = []
Gaussian4 = []
uniform4 = []
Gaussian5 = []
uniform5 = []
Gaussian6 = []
uniform6 = []
Gaussian7 = []
uniform7 = []
    
        

# values = [0, 0.15, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

values = np.arange(0, 1.05, 0.05)


for value in values:
    data = generateRawData(500, 150, value, 'uniform')
    
    # EXP ONE
    uniform.append(runClassifier(data, 'SVM').iloc[0, 3])
    uniform5.append(superFunction(data, 'gausNoise', 200, 10, feature_cols = feature_cols,
                                  target = 150, split = 500, classifier = 'SVM', noise = 0.05).iloc[0, 3])
    
    # EXP TWO
    x = data.iloc[:, :2]
    y = data[data.shape[1] - 1]
    xy = pd.concat([x, y], axis = 1, ignore_index=True)
    uniform2.append(runClassifier(xy, 'SVM').iloc[0, 3])
    
    augment = applyAugmentationMethod(data, 'gausNoise', 200, 10, noise = 0.05)
    log = logReg(dataset = augment, feature_cols = feature_cols, target = 150, split = 500)
    a = log.iloc[:, :2]
    b = log[data.shape[1] - 1]
    ab = pd.concat([a, b], axis = 1, ignore_index=True)
    uniform6.append(runClassifier(ab, 'SVM').iloc[0, 3])
    
    
    #EXP THREE
    data2 = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    X = pca.fit_transform(data2)
    Z = pd.DataFrame(X)
    XY = pd.concat([Z, y], axis = 1, ignore_index=True)
    uniform3.append(runClassifier(XY, 'SVM').iloc[0, 3])
    
    augment2 = applyAugmentationMethod(data, 'gausNoise', 200, 10, noise = 0.05)
    log2 = logReg(dataset = augment2, feature_cols = feature_cols, target = 150, split = 500)
    label2 = log2[data.shape[1] - 1]
    data3 = StandardScaler().fit_transform(log2)
    pca = PCA(n_components=2)
    A = pca.fit_transform(data3)
    B = pd.DataFrame(A)
    AB = pd.concat([B, label2], axis = 1, ignore_index=True)
    uniform7.append(runClassifier(AB, 'SVM').iloc[0, 3])
    
for value in values:
    data = generateRawData(500, 150, value, 'gaussian')
    
    # EXP ONE
    Gaussian.append(runClassifier(data, 'SVM').iloc[0, 3])
    Gaussian5.append(superFunction(data, 'gausNoise', 200, 10, feature_cols = feature_cols,
                                  target = 150, split = 500, classifier = 'SVM', noise = 0.05).iloc[0, 3])
    
    # EXP TWO
    x = data.iloc[:, :2]
    y = data[data.shape[1] - 1]
    xy = pd.concat([x, y], axis = 1, ignore_index=True)
    Gaussian2.append(runClassifier(xy, 'SVM').iloc[0, 3])
    
    augment = applyAugmentationMethod(data, 'gausNoise', 200, 10, noise = 0.05)
    log = logReg(dataset = augment, feature_cols = feature_cols, target = 150, split = 500)
    a = log.iloc[:, :2]
    b = log[data.shape[1] - 1]
    ab = pd.concat([a, b], axis = 1, ignore_index=True)
    Gaussian6.append(runClassifier(ab, 'SVM').iloc[0, 3])
    
    
    #EXP THREE
    data2 = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    X = pca.fit_transform(data2)
    Z = pd.DataFrame(X)
    XY = pd.concat([Z, y], axis = 1, ignore_index=True)
    Gaussian3.append(runClassifier(XY, 'SVM').iloc[0, 3])
    
    # augment2 = applyAugmentationMethod(data, 'gausNoise', 200, 10, noise = 0.05)
    # log2 = logReg(dataset = augment2, feature_cols = feature_cols, target = 150, split = 500)
    # data3 = StandardScaler().fit_transform(log2)
    # pca = PCA(n_components=2)
    # A = pca.fit_transform(data3)
    # B = pd.DataFrame(A)
    # y = data[data.shape[1] - 1]
    # AB = pd.concat([B, y], axis = 1, ignore_index=True)
    # Gaussian7.append(runClassifier(AB, 'SVM').iloc[0, 3])
    
fig, ax = plt.subplots(4, 3, sharey = True, figsize = (20, 10))

ax[0, 0].plot(values, uniform, marker = 'o')
ax[0, 0].set_title("500 by 150")
ax[0, 0].set_ylabel("Accuracy")
ax[0, 0].grid()

ax[0, 1].plot(values, uniform2, marker = 'o')
ax[0, 1].set_title("First Two Cols")
ax[0, 1].grid()

ax[0, 2].plot(values, uniform3, marker = 'o')
ax[0, 2].set_title("PCA")
ax[0, 2].grid()

ax[1, 0].plot(values, Gaussian, marker = 'o')
# ax[1, 0].set_title("Raw Gaussian")
ax[1, 0].set_ylabel("Accuracy")
ax[1, 0].grid()

ax[1, 1].plot(values, Gaussian2, marker = 'o')
# ax[1, 1].set_title("First Two Cols")
ax[1, 1].grid()

ax[1, 2].plot(values, Gaussian3, marker = 'o')
# ax[1, 2].set_title("PCA")
ax[1, 2].grid()

ax[2, 0].plot(values, uniform5, marker = 'o')
# ax[0, 0].set_title("Raw Uniform")
ax[2, 0].set_ylabel("Accuracy")
ax[2, 0].grid()

ax[2, 1].plot(values, uniform6, marker = 'o')
# ax[0, 1].set_title("First Two Cols")
ax[2, 1].grid()

ax[2, 2].plot(values, uniform7, marker = 'o')
ax[2,2].grid()
# ax[0, 2].set_title("PCA")

ax[3, 0].plot(values, Gaussian5, marker = 'o')
# ax[1, 0].set_title("Raw Gaussian")
ax[3, 0].set_ylabel("Accuracy")
ax[3, 0].grid()

ax[3, 1].plot(values, Gaussian6, marker = 'o')
# ax[1, 1].set_title("First Two Cols")
ax[3, 1].grid()

# ax[3, 2].plot(values, Gaussian7, marker = 'o')
# # ax[1, 2].set_title("PCA")
# ax[3, 2].grid()

plt.tight_layout()
plt.show()

plt.plot(values, uniform, marker = 'o', label = '500 by 150', color = 'magenta')
plt.plot(values, uniform2, marker = 'o', label = '500 by 2', color = 'black')
plt.plot(values, uniform3, marker = 'o', label = 'PCA 500 by 2', color = 'blue')
plt.title('Uniform Raw Accuracy')
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Cluster Distance')
plt.show()

plt.plot(values, uniform, marker = 'o', label = 'Raw 500 by 150', color = 'black')
plt.plot(values, uniform5, marker = 'o', label = 'Augmented 500 by 150', color = 'blue')
plt.legend()
plt.title('Raw vs Augmented')
plt.ylabel('Accuracy')
plt.xlabel('Cluster Distance')
plt.show()

plt.plot(values, uniform2, marker = 'o', label = 'Raw 500 by 2', color = 'black')
plt.plot(values, uniform6, marker = 'o', label = 'Augmented 500 by 2', color = 'blue')
plt.title('Raw vs Augmented')
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Cluster Distance')
plt.show()

plt.plot(values, uniform3, marker = 'o', label = 'Raw PCA 500 by 2', color = 'black')
plt.plot(values, uniform7, marker = 'o', label = 'Augmented PCA 500 by 2', color = 'blue')
plt.title('Raw vs Augmented')
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Cluster Distance')
plt.show()
