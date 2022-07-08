# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:46:35 2022

@author: cdiet
"""
from fakeSuper import superFunction
import matplotlib.pyplot as plt
from generateRawData import generateRawData
from fakeSuper import runClassifier



feature_cols = []
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
    

values = [0, 0.15, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


for value in values:
    data = generateRawData(500, 150, value, 'uniform')
    
    uniform.append(runClassifier(data, 'SVM').iloc[0, 3])
    #pmOne
    uniform2.append(superFunction(data, "pmOne", 200, 30, feature_cols = feature_cols,
                  target = 150, split = 500, classifier = 'SVM', unit = 0.1).iloc[0, 3])

    #randSwap
    uniform3.append(superFunction(data, "randSwap", 200, 30, feature_cols = feature_cols,
                  target = 150, split = 500, classifier = 'SVM', unit = 0.1).iloc[0, 3])

    #gausNoise
    uniform4.append(superFunction(data, "gausNoise", 200, 30, feature_cols = feature_cols,
                  target = 150, split = 500, classifier = 'SVM', noise = 0.05).iloc[0, 3])
  
    
for value in values:
    data = generateRawData(500, 150, value, 'gaussian')
    
    Gaussian.append(runClassifier(data, 'SVM').iloc[0, 3])
    #pmOne
    Gaussian2.append(superFunction(data, "pmOne", 200, 30, feature_cols = feature_cols,
                  target = 150, split = 500, classifier = 'SVM', unit = 0.1).iloc[0, 3])

    #randSwap
    Gaussian3.append(superFunction(data, "randSwap", 200, 30, feature_cols = feature_cols,
                  target = 150, split = 500, classifier = 'SVM').iloc[0, 3])

    #gausNoise
    Gaussian4.append(superFunction(data, "gausNoise", 200, 30, feature_cols = feature_cols,
                  target = 150, split = 500, classifier = 'SVM', noise = 0.05).iloc[0, 3])



fig, ax = plt.subplots(2, 2, sharey = True)

ax[0, 0].plot(values, Gaussian, marker = 'd')
ax[0, 0].set_title("Gaussian Raw Data Accuracy")
ax[0, 0].set_ylabel("Accuracy")

ax[0, 1].plot(values, uniform, marker = 'd')
ax[0, 1].set_title("Uniform Raw Data Accuracy")

ax[1, 0].plot(values, Gaussian2, marker = 'd')
ax[1, 0].set_title("Gaussian Augmented Data Accuracy")
ax[1, 0].set_ylabel("Accuracy")
ax[1, 0].set_xlabel("Distance Between Clusters")

ax[1, 1].plot(values, uniform2, marker = 'd')
ax[1, 1].set_title("Uniform Augmented Data Accuracy")
ax[1, 1].set_xlabel("Distance Between Clusters")

plt.tight_layout()

# # RANDSWAP

fig, ax2 = plt.subplots(2, 2, sharey = True)

ax2[0, 0].plot(values, Gaussian, marker = 'd')
ax2[0, 0].set_title("Gaussian Raw Data Accuracy")
ax2[0, 0].set_ylabel("Accuracy")


ax2[0, 1].plot(values, uniform, marker = 'd')
ax2[0, 1].set_title("Uniform Raw Data Accuracy")

ax2[1, 0].plot(values, Gaussian3, marker = 'd')
ax2[1, 0].set_title("Gaussian Augmented Data Accuracy")
ax2[1, 0].set_ylabel("Accuracy")
ax2[1, 0].set_xlabel("Distance Between Clusters")

ax2[1, 1].plot(values, uniform3, marker = 'd')
ax2[1, 1].set_title("Uniform Augmented Data Accuracy")
ax2[1, 1].set_xlabel("Distance Between Clusters")

plt.tight_layout()

# # GAUSNOISE

fig, ax3 = plt.subplots(2, 2, sharey = True)

ax3[0, 0].plot(values, Gaussian, marker = 'd')
ax3[0, 0].set_title("Gaussian Raw Data Accuracy")
ax3[0, 0].set_ylabel("Accuracy")

ax3[0, 1].plot(values, uniform, marker = 'd')
ax3[0, 1].set_title("Uniform Raw Data Accuracy")

ax3[1, 0].plot(values, Gaussian4, marker = 'd')
ax3[1, 0].set_title("Gaussian Augment Data Accuracy")
ax3[1, 0].set_ylabel("Accuracy")
ax3[1, 0].set_xlabel("Distance Between Clusters")

ax3[1, 1].plot(values, uniform4, marker = 'd')
ax3[1, 1].set_title("Uniform Augment Data Accuracy")
ax3[1, 1].set_xlabel("Distance Between Clusters")


plt.tight_layout()
