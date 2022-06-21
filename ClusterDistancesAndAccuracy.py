# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:37:01 2022

@author: cdiet
"""

import matplotlib.pyplot as plt
from FunctionReturnsF1 import OkayFunction
from applyAugmentationMethod import applyAugmentationMethod
from LogisticRegressionReal import LogReg


Gaussian = []
distancesGaussian = [0.5, 1.0, 1.5, 2.0]
uniform = []
distancesUniform = [0.15, 0.6, 1.0, 2.0]

Gaussian.append(OkayFunction('Gaussian_Data_0.5_Unit.txt'))
Gaussian.append(OkayFunction('Gaussian_Data_1.0_Unit.txt'))
Gaussian.append(OkayFunction('Gaussian_Data_1.5_Unit.txt'))
Gaussian.append(OkayFunction('Generated Gaussian Distribution.txt'))
uniform.append(OkayFunction('Uniform_Data_0.6_Unit.txt'))
uniform.append(OkayFunction('Uniform_Data_1.0_Unit.txt'))
uniform.append(OkayFunction('Uniform_Data_0.15_Unit.txt'))
uniform.append(OkayFunction('synthetic_data_with_labels.txt'))

feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)
   
Gaussian2 = []

uniform2 = []

Gaussian2.append(OkayFunction((LogReg((applyAugmentationMethod(
    'Gaussian_Data_0.5_Unit.txt', "pmOne", 200, 30, unit = 0.1)),
    feature_cols = feature_cols, target= 150, split = 500))))
Gaussian2.append(OkayFunction((LogReg((applyAugmentationMethod(
    'Gaussian_Data_1.0_Unit.txt', 'pmOne', 200, 30, unit = 0.1)),
    feature_cols = feature_cols, target= 150, split = 500))))
Gaussian2.append(OkayFunction((LogReg((applyAugmentationMethod(
    'Gaussian_Data_1.5_Unit.txt', 'pmOne', 200, 30, unit = 0.1)),
    feature_cols = feature_cols, target= 150, split = 500))))
Gaussian2.append(OkayFunction((LogReg((applyAugmentationMethod(
    'Generated Gaussian Distribution.txt', 'pmOne', 200, 30, unit = 0.1)),
    feature_cols = feature_cols, target= 150, split = 500))))
uniform2.append(OkayFunction((LogReg((applyAugmentationMethod(
    'Uniform_Data_0.6_Unit.txt', 'pmOne', 200, 30, unit = 0.1)), 
    feature_cols = feature_cols, target= 150, split = 500))))
uniform2.append(OkayFunction((LogReg((applyAugmentationMethod(
    'Uniform_Data_1.0_Unit.txt', 'pmOne', 200, 30, unit = 0.1)),
    feature_cols = feature_cols, target= 150, split = 500))))
uniform2.append(OkayFunction((LogReg((applyAugmentationMethod(
    'Uniform_Data_0.15_Unit.txt', 'pmOne', 200, 30, unit = 0.1)),
    feature_cols = feature_cols, target= 150, split = 500))))
uniform2.append(OkayFunction((LogReg((applyAugmentationMethod(
    'synthetic_data_with_labels.txt', 'pmOne', 200, 30, unit = 0.1)),
    feature_cols = feature_cols, target= 150, split = 500))))



fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(distancesGaussian, Gaussian)
ax[0, 0].set_title("Gaussian Raw Data Accuracy")
ax[0, 0].set(xlabel = None)

ax[0, 1].plot(distancesUniform, uniform)
ax[0, 1].set_title("Uniform Raw Data Accuracy")

ax[1, 0].plot(distancesGaussian, Gaussian2)
ax[1, 0].set_title("Gaussian Augmented Data Accuracy")
ax[1, 0].set(xlabel = None)

ax[1, 1].plot(distancesUniform, uniform2)
ax[1, 1].set_title("Uniform Augmented Data Accuracy")