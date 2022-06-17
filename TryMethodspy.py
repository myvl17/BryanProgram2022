# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 13:56:58 2022

@author: cdiet
"""

from UniformAugmentation import RandUnit

# Uniform +- 1 unit
print(RandUnit('synthetic_data_with_labels.txt', 500, 0.1))

# Gaussian +- 1 unit 
print(RandUnit('Generated Gaussian Distribution.txt', 500, 0.1))