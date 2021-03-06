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

# Empty lists for plots
Gaussian = []
uniform = []
Gaussian2 = []
uniform2 = []
Gaussian3 = []
uniform3 = []
Gaussian4 = []
uniform4 = []

# Ticks for plots
distancesGaussian = [10**-30, 0.0000000001, 0.0001, 0.05, 0.5, 1.0, 1.5, 2.0]
distancesUniform = [10**-30, 0.0000000001, 0.0001, 0.05, 0.15, 0.6, 1.0, 2.0]

feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)

# filesG = ["Gaussian_Data_0.05_Unit.txt", 'Gaussian_Data_0.5_Unit.txt',
#          'Gaussian_Data_1.0_Unit.txt', 'Gaussian_Data_1.5_Unit.txt',
#          'Generated Gaussian Distribution.txt', 'Gaussian_Data_0.0001_Unit.txt',
#          'Gaussian_Data_-5_Unit.txt', 'Gaussian_Data_-10_Unit.txt']
# filesU = ["Uniform_Data_0.05_Unit.txt", 'Uniform_Data_0.6_Unit.txt',
#          'Uniform_Data_1.0_Unit.txt', 'Uniform_Data_0.15_Unit.txt',
#          'synthetic_data_with_labels.txt', 'Uniform_Data_0.0001_Unit.txt',
#          'Uniform_Data_10e-10_Unit.txt', 'Uniform_Data_10e-30_Unit.txt']

filesU = ['Uniform_Data_10e-30_Unit.txt']

# for file in filesG:
#     Gaussian.append(OkayFunction(file))
#     #pmOne
    # Gaussian2.append(OkayFunction((LogReg((applyAugmentationMethod(
    #     file, "pmOne", 200, 30, unit = 0.1)),
    #     feature_cols = feature_cols, target= 150, split = 500))))
    # #randSwap
    # Gaussian3.append(OkayFunction((LogReg((applyAugmentationMethod(
    #     file, "randSwap", 200, 30, unit = 0.1)),
    #     feature_cols = feature_cols, target= 150, split = 500))))
    # #gausNoise
    # Gaussian4.append(OkayFunction((LogReg((applyAugmentationMethod(
    #     file, "pmOne", 200, 30, unit = 0.1)),
    #     feature_cols = feature_cols, target= 150, split = 500))))
    
for file in filesU:
    uniform.append(OkayFunction(file))
    #pmOne
    # uniform2.append(OkayFunction((LogReg((applyAugmentationMethod(
    #     file, "pmOne", 200, 30, unit = 0.1)),
    #     feature_cols = feature_cols, target= 150, split = 500))))
    # #randSwap
    # uniform3.append(OkayFunction((LogReg((applyAugmentationMethod(
    #     file, "randSwap", 200, 30, unit = 0.1)),
    #     feature_cols = feature_cols, target= 150, split = 500))))
    # #gausNoise
    # uniform4.append(OkayFunction((LogReg((applyAugmentationMethod(
    #     file, 'gausNoise', 200, 30, noise = 0.05)), 
    #     feature_cols = feature_cols, target= 150, split = 500))))


# fig, ax = plt.subplots(2, 2)

# ax[0, 0].plot(distancesGaussian, Gaussian)
# ax[0, 0].set_title("Gaussian Raw Data Accuracy")
# ax[0, 0].set_ylabel("Accuracy")

# ax[0, 1].plot(distancesUniform, uniform)
# ax[0, 1].set_title("Uniform Raw Data Accuracy")

# ax[1, 0].plot(distancesGaussian, Gaussian2)
# ax[1, 0].set_title("Gaussian Augmented Data Accuracy")
# ax[1, 0].set_ylabel("Accuracy")
# ax[1, 0].set_xlabel("Distance Between Clusters")

# ax[1, 1].plot(distancesUniform, uniform2)
# ax[1, 1].set_title("Uniform Augmented Data Accuracy")
# ax[1, 1].set_xlabel("Distance Between Clusters")

# plt.tight_layout()

# # # RANDSWAP

# fig, ax2 = plt.subplots(2, 2)

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

# # # GAUSNOISE

# fig, ax3 = plt.subplots(2, 2)

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
