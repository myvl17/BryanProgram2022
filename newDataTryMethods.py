# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:44:40 2022

@author: cdiet
"""

import numpy as np

from generateRawData import generateRawData
np.savetxt("Uniform_Data_1.0_Unit.txt", generateRawData(500, 7, 1.0, "uniform"))

np.savetxt("Uniform_Data_2.0_Unit.txt", generateRawData(500, 7, 2.0, "uniform"))

np.savetxt("Uniform_Data_0.5_Unit.txt", generateRawData(500, 7, 0.5, "uniform"))

np.savetxt("Uniform_Data_1.5_Unit.txt", generateRawData(500, 7, 1.5, "uniform"))

np.savetxt("Gaussian_Data_1.5_Unit.txt", generateRawData(500, 7, 1.5, "gaussian"))

np.savetxt("Gaussian_Data_1.0_Unit.txt", generateRawData(500, 7, 1.0, "gaussian"))

np.savetxt("Gaussian_Data_0.5_Unit.txt", generateRawData(500, 7, 0.5, "gaussian"))

np.savetxt("Gaussian_Data_0_Unit.txt", generateRawData(500, 7, 0, "gaussian"))

np.savetxt("Uniform_Data_0_Unit.txt", generateRawData(500, 7, 0, "uniform"))

np.savetxt("Gaussian_Data_2.0_Unit.txt", generateRawData(500, 7, 2.0, "gaussian"))

np.savetxt("Uniform_Data_0.75_Unit.txt", generateRawData(500, 7, 0.75, "uniform"))

np.savetxt("Gaussian_Data_0.75_Unit.txt", generateRawData(500, 7, 0.75, "gaussian"))

np.savetxt("Uniform_Data_0.90_Unit.txt", generateRawData(500, 7, 0.90, "uniform"))

np.savetxt("Gaussian_Data_0.90_Unit.txt", generateRawData(500, 7, 0.90, "gaussian"))

np.savetxt("Uniform_Data_1.1_Unit.txt", generateRawData(500, 7, 1.1, "uniform"))

np.savetxt("Gaussian_Data_1.1_Unit.txt", generateRawData(500, 7, 1.25, "gaussian"))

np.savetxt("Uniform_Data_1.25_Unit.txt", generateRawData(500, 7, 1.25, "uniform"))

np.savetxt("Gaussian_Data_1.25_Unit.txt", generateRawData(500, 7, 1.1, "gaussian"))



from applyAugmentationMethod import applyAugmentationMethod

# applyAugmentationMethod('Generated Gaussian Distribution.txt', "randSwap", 200, 30)

# applyAugmentationMethod('Generated Gaussian Distribution.txt', "pmOne", 200, 30, unit = 0.1)

# applyAugmentationMethod('Generated Gaussian Distribution.txt', "gausNoise", 200, 30, noise = 0.05)

# applyAugmentationMethod('synthetic_data_with_labels.txt', "randSwap", 200, 30)

# applyAugmentationMethod('synthetic_data_with_labels.txt', "pmOne", 200, 30, unit = 0.1)

# applyAugmentationMethod('synthetic_data_with_labels.txt', "gausNoise", 200, 30, noise = 0.05)

