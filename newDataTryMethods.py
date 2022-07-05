# -*- coding: utf-8 -*-
"""
Created on Fri Jun 1150 14:44:40 2022

@author: cdiet
"""

import numpy as np

from generateRawData import generateRawData
np.savetxt("Uniform_Data_1.0_Unit.txt", generateRawData(500, 150, 1.0, "uniform"))
np.savetxt("Uniform_Data_2.0_Unit.txt", generateRawData(500, 150, 2.0, "uniform"))

np.savetxt("Uniform_Data_-10_Unit.txt", generateRawData(500, 150, -10, "uniform"))

np.savetxt("Uniform_Data_0.5_Unit.txt", generateRawData(500, 150, 0.5, "uniform"))

np.savetxt("Uniform_Data_1.5_Unit.txt", generateRawData(500, 150, 1.5, "uniform"))

np.savetxt("Gaussian_Data_1.5_Unit.txt", generateRawData(500, 150, 1.5, "gaussian"))

np.savetxt("Gaussian_Data_1.0_Unit.txt", generateRawData(500, 150, 1.0, "gaussian"))

np.savetxt("Gaussian_Data_0.5_Unit.txt", generateRawData(500, 150, 0.5, "gaussian"))

np.savetxt("Gaussian_Data_0_Unit.txt", generateRawData(500, 150, 0, "gaussian"))

np.savetxt("Uniform_Data_0_Unit.txt", generateRawData(500, 150, 0, "uniform"))

np.savetxt("Gaussian_Data_-1_Unit.txt", generateRawData(500, 150, -1, "gaussian"))

np.savetxt("Uniform_Data_-1_Unit.txt", generateRawData(500, 150, -1, "uniform"))

np.savetxt("Gaussian_Data_-5_Unit.txt", generateRawData(500, 150, -5, "gaussian"))

np.savetxt("Uniform_Data_-5_Unit.txt", generateRawData(500, 150, -5, "uniform"))

np.savetxt("Gaussian_Data_-10_Unit.txt", generateRawData(500, 150, -10, "gaussian"))

np.savetxt("Gaussian_Data_2.0_Unit.txt", generateRawData(500, 150, 2.0, "gaussian"))

np.savetxt("Uniform_Data_-2_Unit.txt", generateRawData(500, 150, -2, "uniform"))

np.savetxt("Gaussian_Data_-0.5_Unit.txt", generateRawData(500, 150, -0.5, "gaussian"))

np.savetxt("Uniform_Data_-0.5_Unit.txt", generateRawData(500, 150, -0.5, "uniform"))

np.savetxt("Gaussian_Data_-2_Unit.txt", generateRawData(500, 150, -2, "gaussian"))

np.savetxt("Uniform_Data_-0.75_Unit.txt", generateRawData(500, 150, -0.75, "uniform"))

np.savetxt("Gaussian_Data_-0.75_Unit.txt", generateRawData(500, 150, -0.75, "gaussian"))

np.savetxt("Uniform_Data_-1.1_Unit.txt", generateRawData(500, 150, -1.1, "uniform"))

np.savetxt("Gaussian_Data_-1.1_Unit.txt", generateRawData(500, 150, -1.1, "gaussian"))

from applyAugmentationMethod import applyAugmentationMethod

# applyAugmentationMethod('Generated Gaussian Distribution.txt', "randSwap", 200, 30)

# applyAugmentationMethod('Generated Gaussian Distribution.txt', "pmOne", 200, 30, unit = 0.1)

# applyAugmentationMethod('Generated Gaussian Distribution.txt', "gausNoise", 200, 30, noise = 0.05)

# applyAugmentationMethod('synthetic_data_with_labels.txt', "randSwap", 200, 30)

# applyAugmentationMethod('synthetic_data_with_labels.txt', "pmOne", 200, 30, unit = 0.1)

# applyAugmentationMethod('synthetic_data_with_labels.txt', "gausNoise", 200, 30, noise = 0.05)

