# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 21:43:30 2022

@author: cdiet
"""

from fakeSuper import superFunction
import matplotlib.pyplot as plt
from generateRawData import generateRawData
from superFunction import runClassifier
from superFunction import applyAugmentationMethod

feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)

# superFunction("Gaussian_Data_2.0_Unit.txt", "pmOne", 200, 30, feature_cols = feature_cols,
#               target = 150, split = 500, classifier = 'SVM', unit = 0.1)

# plt.show()

# superFunction("uniform_Data_0.5_Unit.txt", "pmOne", 200, 30, feature_cols = feature_cols,
#               target = 150, split = 500, classifier = 'SVM', unit = 0.1)

test = generateRawData(500, 150, 0.75, 'uniform')
 
test2 = (superFunction(file = test, method = "gausNoise", nrows = 200, nvalues = 30, feature_cols = feature_cols,
              target = 150, split = 500, classifier = 'SVM', noise = 0.05).iloc[0, 3])

# test2 = (superFunction(file = test, method = "gausNoise", nrows = 200, nvalues = 30, feature_cols = feature_cols,
#               target = 150, split = 500, classifier = 'SVM', noise = 0.05).iloc[0, 3])

# import pandas as pd

# x = [.25, .75, .4, .8, .9, 1, 1.1, .95, .925, .85]
# y = [.3, .2, .75, .6, .75, .5, .75, .9, 1, 1.1]
 
# labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
 
# df = pd.DataFrame({0:x, 1:y, 2:labels})

# test = applyAugmentationMethod(df, 'gausNoise', 8, 0, noise = 0.05)

