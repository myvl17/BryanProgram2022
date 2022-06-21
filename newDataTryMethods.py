# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:44:40 2022

@author: cdiet
"""

import numpy as np

from generateRawData import generateRawData
np.savetxt("Uniform_Data_1.0_Unit.txt", generateRawData(500, 150, 1.5, "uniform"))

np.savetxt("Uniform_Data_0.6_Unit.txt", generateRawData(500, 150, 0.6, "uniform"))

np.savetxt("Uniform_Data_0.15_Unit.txt", generateRawData(500, 150, 0.15, "uniform"))

np.savetxt("Gaussian_Data_1.5_Unit.txt", generateRawData(500, 150, 1.5, "gaussian"))

np.savetxt("Gaussian_Data_1.0_Unit.txt", generateRawData(500, 150, 1.0, "gaussian"))

np.savetxt("Gaussian_Data_0.5_Unit.txt", generateRawData(500, 150, 0.5, "gaussian"))

from applyAugmentationMethod import applyAugmentationMethod

applyAugmentationMethod('Generated Gaussian Distribution.txt', "randSwap", 200, 30)

applyAugmentationMethod('Generated Gaussian Distribution.txt', "pmOne", 200, 30, unit = 0.1)

applyAugmentationMethod('Generated Gaussian Distribution.txt', "gausNoise", 200, 30, noise = 0.05)

applyAugmentationMethod('synthetic_data_with_labels.txt', "randSwap", 200, 30)

applyAugmentationMethod('synthetic_data_with_labels.txt', "pmOne", 200, 30, unit = 0.1)

applyAugmentationMethod('synthetic_data_with_labels.txt', "gausNoise", 200, 30, noise = 0.05)
