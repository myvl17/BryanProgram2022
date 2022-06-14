# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:20:34 2022

@author: jeffr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(1)

df0 = pd.DataFrame(np.random.normal(5, 1, size=(250, 150)))
df0["Status"] = 0
df1 = pd.DataFrame(np.random.normal(15, 1, size=(250, 150)))
df1["Status"] = 1


dataset = pd.concat([df0, df1])
df = pd.DataFrame(np.random.permutation(dataset))


plt.scatter(df[0], df[1])
plt.show()

np.savetxt('Gaussian Distribution Data Set with Status.txt', df)