# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 21:43:30 2022

@author: cdiet
"""

from superFunction import superFunction
import matplotlib.pyplot as plt

feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)

superFunction("Gaussian_Data_2.0_Unit.txt", "pmOne", 200, 30, feature_cols = feature_cols,
              target = 150, split = 500, classifier = 'SVM', unit = 0.1)

plt.show()

superFunction("uniform_Data_0_Unit.txt", "pmOne", 200, 30, feature_cols = feature_cols,
              target = 150, split = 500, classifier = 'SVM', unit = 0.1)
