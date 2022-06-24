# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:37:01 2022

@author: cdiet
"""

import matplotlib.pyplot as plt
from FunctionReturnsF1 import OkayFunction
from applyAugmentationMethod import applyAugmentationMethod
from LogisticRegressionReal import LogReg
import random

random.seed(1)

Gaussian = []
distancesGaussian = [0.05, 0.5, 1.0, 1.5, 2.0]
uniform = []
distancesUniform = [0.05, 0.15, 0.6, 1.0, 2.0]

Gaussian.append(OkayFunction('Gaussian_Data_0.05_Unit.txt'))
Gaussian.append(OkayFunction('Gaussian_Data_0.5_Unit.txt'))
Gaussian.append(OkayFunction('Gaussian_Data_1.0_Unit.txt'))
Gaussian.append(OkayFunction("Gaussian_Data_0.05_Unit.txt"))
Gaussian.append(OkayFunction('Generated Gaussian Distribution.txt'))
uniform.append(OkayFunction('Uniform_Data_0.05_Unit.txt'))
uniform.append(OkayFunction('Uniform_Data_0.6_Unit.txt'))
uniform.append(OkayFunction('Uniform_Data_1.0_Unit.txt'))
uniform.append(OkayFunction('Uniform_Data_0.15_Unit.txt'))
uniform.append(OkayFunction('synthetic_data_with_labels.txt'))

feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)
   
    
# PMONE
Gaussian2 = []

uniform2 = []

Gaussian2.append(OkayFunction((LogReg((applyAugmentationMethod(
    "Gaussian_Data_0.05_Unit.txt", "pmOne", 200, 30, unit = 0.1)),
    feature_cols = feature_cols, target= 150, split = 500))))
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
    'Uniform_Data_0.05_Unit.txt', 'pmOne', 200, 30, unit = 0.1)), 
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

# RANDSWAP

# fig, ax2 = plt.subplots(2, 2)
   
# Gaussian3 = []

# uniform3 = []

# Gaussian3.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Gaussian_Data_0.5_Unit.txt', "randSwap", 200, 30, unit = 0.1)),
#     feature_cols = feature_cols, target= 150, split = 500))))
# Gaussian3.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Gaussian_Data_1.0_Unit.txt', 'randSwap', 200, 30, unit = 0.1)),
#     feature_cols = feature_cols, target= 150, split = 500))))
# Gaussian3.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Gaussian_Data_1.5_Unit.txt', 'randSwap', 200, 30, unit = 0.1)),
#     feature_cols = feature_cols, target= 150, split = 500))))
# Gaussian3.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Generated Gaussian Distribution.txt', 'randSwap', 200, 30, unit = 0.1)),
#     feature_cols = feature_cols, target= 150, split = 500))))
# uniform3.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Uniform_Data_0.6_Unit.txt', 'randSwap', 200, 30, unit = 0.1)), 
#     feature_cols = feature_cols, target= 150, split = 500))))
# uniform3.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Uniform_Data_1.0_Unit.txt', 'randSwap', 200, 30, unit = 0.1)),
#     feature_cols = feature_cols, target= 150, split = 500))))
# uniform3.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Uniform_Data_0.15_Unit.txt', 'randSwap', 200, 30, unit = 0.1)),
#     feature_cols = feature_cols, target= 150, split = 500))))
# uniform3.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'synthetic_data_with_labels.txt', 'randSwap', 200, 30, unit = 0.1)),
#     feature_cols = feature_cols, target= 150, split = 500))))

# ax2[0, 0].plot(distancesGaussian, Gaussian)
# ax2[0, 0].set_title("Gaussian Raw Data Accuracy")
# ax2[0, 0].set_ylabel("Accuracy")

# ax2[0, 1].plot(distancesUniform, uniform)
# ax2[0, 1].set_title("Uniform Raw Data Accuracy")

# ax2[1, 0].plot(distancesGaussian, Gaussian3)
# ax2[1, 0].set_title("Gaussian Augmented Data Accuracy")
# ax2[1, 0].set_ylabel("Accuracy")
# ax2[1, 0].set_xlabel("Distance Between Clusters")

# ax2[1, 1].plot(distancesUniform, uniform3)
# ax2[1, 1].set_title("Uniform Augmented Data Accuracy")
# ax2[1, 1].set_xlabel("Distance Between Clusters")

# plt.tight_layout()

# # GAUSNOISE

# fig, ax3 = plt.subplots(2, 2)

# Gaussian4 = []

# uniform4 = []

# Gaussian4.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Gaussian_Data_0.5_Unit.txt', "gausNoise", 200, 30, noise = 0.05)),
#     feature_cols = feature_cols, target= 150, split = 500))))
# Gaussian4.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Gaussian_Data_1.0_Unit.txt', 'gausNoise', 200, 30, noise = 0.05)),
#     feature_cols = feature_cols, target= 150, split = 500))))
# Gaussian4.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Gaussian_Data_1.5_Unit.txt', 'gausNoise', 200, 30, noise = 0.05)),
#     feature_cols = feature_cols, target= 150, split = 500))))
# Gaussian4.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Generated Gaussian Distribution.txt', 'gausNoise', 200, 30, noise = 0.05)),
#     feature_cols = feature_cols, target= 150, split = 500))))
# uniform4.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Uniform_Data_0.6_Unit.txt', 'gausNoise', 200, 30, noise = 0.05)), 
#     feature_cols = feature_cols, target= 150, split = 500))))
# uniform4.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Uniform_Data_1.0_Unit.txt', 'gausNoise', 200, 30,noise = 0.05)),
#     feature_cols = feature_cols, target= 150, split = 500))))
# uniform4.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'Uniform_Data_0.15_Unit.txt', 'gausNoise', 200, 30, noise = 0.05)),
#     feature_cols = feature_cols, target= 150, split = 500))))
# uniform4.append(OkayFunction((LogReg((applyAugmentationMethod(
#     'synthetic_data_with_labels.txt', 'gausNoise', 200, 30, noise = 0.05)),
#     feature_cols = feature_cols, target= 150, split = 500))))

# ax3[0, 0].plot(distancesGaussian, Gaussian)
# ax3[0, 0].set_title("Gaussian Raw Data Accuracy")
# ax3[0, 0].set_ylabel("Accuracy")

# ax3[0, 1].plot(distancesUniform, uniform)
# ax3[0, 1].set_title("Uniform Raw Data Accuracy")

# ax3[1, 0].plot(distancesGaussian, Gaussian4)
# ax3[1, 0].set_title("Gaussian Augment Data Accuracy")
# ax3[1, 0].set_ylabel("Accuracy")
# ax3[1, 0].set_xlabel("Distance Between Clusters")

# ax3[1, 1].plot(distancesUniform, uniform4)
# ax3[1, 1].set_title("Uniform Augment Data Accuracy")
# ax3[1, 1].set_xlabel("Distance Between Clusters")

# plt.tight_layout()
