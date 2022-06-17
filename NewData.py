# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:44:40 2022

@author: cdiet
"""

import numpy as np

from generateRawData import generateRawData
np.savetxt("Uniform_Data_1.5_Unit", generateRawData(500, 150, 1.5, "uniform"))

np.savetxt("Uniform_Data_1.0_Unit", generateRawData(500, 150, 1.0, "uniform"))

np.savetxt("Uniform_Data_0.5_Unit", generateRawData(500, 150, 0.5, "uniform"))

np.savetxt("Uniform_Data_1.5_Unit", generateRawData(500, 150, 1.5, "gaussian"))

np.savetxt("Uniform_Data_1.0_Unit", generateRawData(500, 150, 1.0, "gaussian"))

np.savetxt("Uniform_Data_0.5_Unit", generateRawData(500, 150, 0.5, "gaussian"))
