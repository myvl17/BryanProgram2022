# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 13:56:58 2022

@author: cdiet
"""
from applyAugmentationMethod import applyAugmentationMethod

applyAugmentationMethod('Generated Gaussian Distribution.txt', "randSwap", 200, 30)

applyAugmentationMethod('Generated Gaussian Distribution.txt', "pmOne", 200, 30, unit = 0.1)

applyAugmentationMethod('Generated Gaussian Distribution.txt', "gausNoise", 200, 30, noise = 0.05)

applyAugmentationMethod('synthetic_data_with_labels.txt', "randSwap", 200, 30)

applyAugmentationMethod('synthetic_data_with_labels.txt', "pmOne", 200, 30, unit = 0.1)

applyAugmentationMethod('synthetic_data_with_labels.txt', "gausNoise", 200, 30, noise = 0.05)