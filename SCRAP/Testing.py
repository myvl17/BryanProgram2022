# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:46:12 2022

@author: cdiet
"""

from fixingRandomness import applyAugmentationMethod
import pandas as pd
import matplotlib.pyplot as plt

x = [.25, .75, .4, .8, .9, 1, 1.1, .95, .925, .85]
y = [.3, .2, .75, .6, .75, .5, .75, .9, 1, 1.1]
 
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
 
plt.scatter(x, y, c=labels)
plt.show()
df = pd.DataFrame({0:x, 1:y, 2:labels})
 

applyAugmentationMethod(df, method = 'randSwap', nrows = 4, nvalues = 1)

plt.scatter(x, y, c=labels)