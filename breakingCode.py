# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:26:01 2022

@author: cdiet
"""
import numpy as np
import pandas as pd

rawData1 = [0.25, 3.47, 0.74, 0.56, 3.20]
rawData2 = [0.42, 3.85, 0.89, 0.17, 3.98]
labels = [0, 1, 0, 0, 1]



table = pd.DataFrame({'x':rawData1, 'y':rawData2, 'labels':labels})
table = table.reset_index(drop = True)
np.savetxt('rawData.txt', table)

from applyAugmentationMethod import applyAugmentationMethod
from SVC import SVC
from FunctionReturnsF1 import OkayFunction


print(SVC((applyAugmentationMethod('rawData.txt', 'pmOne', 5, 1, unit = 0.1)), 
        [0, 1], 2, 5))