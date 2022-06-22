# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:25:33 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("THIS ONE")

x = [0.25, 0.5, 0.75, 3.25, 3.75]
y = [0.25, 0.75, 0.5, 2.5, 2.25]
labels = [0,0,0,1,1]

df = pd.DataFrame({'x':x, 'y':y, 'labels':labels})
np.savetxt("breaking.txt", df)



from applyAugmentationMethod_distanceMeasuring import applyAugmentationMethod

aug = applyAugmentationMethod(file="breaking.txt", method="gausNoise", nrows=3, nvalues=1, noise=0.1)

from applyAugmentationMethod_distanceMeasuring import LogReg

#logReg = LogReg(aug, [0,1], 2, 5)



plt.scatter(x, y)
plt.yticks(range(0,4,1))
plt.xticks(range(0,4,1))
plt.show()

