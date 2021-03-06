# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:25:41 2022

@author: cdiet
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superFunction import superFunction
from generateRawData import generateRawData
from FunctionReturnsF1 import OkayFunction
from superFunction import runClassifier

# df =  generateRawData(500, 7, -5, "gaussian")

# acc = superFunction(df, "pmOne", 500, 3, feature_cols = [0, 1, 2, 3, 4, 5, 6],
#               target = 7, split = 500, classifier = 'kNN', unit = 0.1)

# Ticks for plots
distancesGaussian = [0, 0.15, 0.25, 0.5, 0.75, 0.90, 1.0, 1.1, 1.25, 1.5, 2.0]
distancesUniform = [0, 0.15, 0.25, 0.5, 0.75, 0.90, 1.0, 1.1, 1.25, 1.5, 2.0]



# Empty lists for plots
Gaussian = []
uniform = []
Gaussian2 = []
uniform2 = []
Gaussian3 = []
uniform3 = []
Gaussian4 = []
uniform4 = []


filesG = ["Gaussian_Data_0_Unit.txt", "Gaussian_Data_0.15_Unit.txt", "Gaussian_Data_0.25_Unit.txt",
         "Gaussian_Data_0.5_Unit.txt", "Gaussian_Data_0.75_Unit.txt", "Gaussian_Data_0.90_Unit.txt",
        "Gaussian_Data_1.0_Unit.txt",
         "Gaussian_Data_1.1_Unit.txt", "Gaussian_Data_1.25_Unit.txt",
          "Gaussian_Data_1.5_Unit.txt", 'Gaussian_Data_2.0_Unit.txt']
filesU = ["Uniform_Data_0_Unit.txt", "Uniform_Data_0.15_Unit.txt", "Uniform_Data_0.25_Unit.txt",
         "Uniform_Data_0.5_Unit.txt", "Uniform_Data_0.75_Unit.txt", "Uniform_Data_0.90_Unit.txt", "Uniform_Data_1.0_Unit.txt",
         "Uniform_Data_1.1_Unit.txt", "Uniform_Data_1.25_Unit.txt",
          "Uniform_Data_1.5_Unit.txt", 'Uniform_Data_2.0_Unit.txt']


feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)


for file in filesG:
    df = pd.read_table(file, delimiter = " ", header = None)
    
    Gaussian.append(runClassifier(df, 'SVM').iloc[0, 3])
    #pmOne
    Gaussian2.append(superFunction(file, "pmOne", 200, 30, feature_cols = feature_cols,
                  target = 150, split = 500, classifier = 'SVM', unit = 0.1).iloc[0, 3])

    #randSwap
    Gaussian3.append(superFunction(file, "randSwap", 200, 30, feature_cols = feature_cols,
                  target = 150, split = 500, classifier = 'SVM').iloc[0, 3])

    #gausNoise
    Gaussian4.append(superFunction(file, "gausNoise", 200, 30, feature_cols = feature_cols,
                  target = 150, split = 500, classifier = 'SVM', noise = 0.05).iloc[0, 3])

    
for file in filesU:
    df = pd.read_table(file, delimiter = " ", header = None)
    
    uniform.append(runClassifier(df, 'SVM').iloc[0, 3])
    #pmOne
    uniform2.append(superFunction(file, "pmOne", 200, 30, feature_cols = feature_cols,
                  target = 150, split = 500, classifier = 'SVM', unit = 0.1).iloc[0, 3])

    #randSwap
    uniform3.append(superFunction(file, "randSwap", 200, 30, feature_cols = feature_cols,
                  target = 150, split = 500, classifier = 'SVM', unit = 0.1).iloc[0, 3])

    #gausNoise
    uniform4.append(superFunction(file, "gausNoise", 200, 30, feature_cols = feature_cols,
                  target = 150, split = 500, classifier = 'SVM', noise = 0.05).iloc[0, 3])


fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(distancesGaussian, Gaussian)
ax[0, 0].set_title("Gaussian Raw Data Accuracy")
ax[0, 0].set_ylabel("Accuracy")

ax[0, 1].plot(distancesUniform, uniform)
ax[0, 1].set_title("Uniform Raw Data Accuracy")

ax[1, 0].plot(distancesGaussian, Gaussian2)
ax[1, 0].set_title("Gaussian Augmented Data Accuracy")
ax[1, 0].set_ylabel("Accuracy")
ax[1, 0].set_xlabel("Distance Between Clusters")

ax[1, 1].plot(distancesUniform, uniform2)
ax[1, 1].set_title("Uniform Augmented Data Accuracy")
ax[1, 1].set_xlabel("Distance Between Clusters")

plt.tight_layout()

# # RANDSWAP

fig, ax2 = plt.subplots(2, 2)

ax2[0, 0].plot(distancesGaussian, Gaussian)
ax2[0, 0].set_title("Gaussian Raw Data Accuracy")
ax2[0, 0].set_ylabel("Accuracy")

ax2[0, 1].plot(distancesUniform, uniform)
ax2[0, 1].set_title("Uniform Raw Data Accuracy")

ax2[1, 0].plot(distancesGaussian, Gaussian3)
ax2[1, 0].set_title("Gaussian Augmented Data Accuracy")
ax2[1, 0].set_ylabel("Accuracy")
ax2[1, 0].set_xlabel("Distance Between Clusters")

ax2[1, 1].plot(distancesUniform, uniform3)
ax2[1, 1].set_title("Uniform Augmented Data Accuracy")
ax2[1, 1].set_xlabel("Distance Between Clusters")

plt.tight_layout()

# # GAUSNOISE

fig, ax3 = plt.subplots(2, 2)

ax3[0, 0].plot(distancesGaussian, Gaussian)
ax3[0, 0].set_title("Gaussian Raw Data Accuracy")
ax3[0, 0].set_ylabel("Accuracy")

ax3[0, 1].plot(distancesUniform, uniform)
ax3[0, 1].set_title("Uniform Raw Data Accuracy")

ax3[1, 0].plot(distancesGaussian, Gaussian4)
ax3[1, 0].set_title("Gaussian Augment Data Accuracy")
ax3[1, 0].set_ylabel("Accuracy")
ax3[1, 0].set_xlabel("Distance Between Clusters")

ax3[1, 1].plot(distancesUniform, uniform4)
ax3[1, 1].set_title("Uniform Augment Data Accuracy")
ax3[1, 1].set_xlabel("Distance Between Clusters")

plt.tight_layout()

