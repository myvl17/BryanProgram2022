# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:24:23 2022

@author: cdiet
"""
#https://linuxhint.com/generating-random-numbers-with-uniform-distribution-in-python/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

np.random.seed(1)

df1 = pd.DataFrame(np.random.uniform(0, 1, (250, 150)))
df2 = pd.DataFrame(np.random.uniform(3, 4, (250, 150)))


df = pd.concat([df1, df2])

perm = np.random.permutation(df)

dfreal = pd.DataFrame(perm)

np.savetxt('test.txt', dfreal)

fig, ax = plt.subplots()

plt.scatter(df[0:249], df[250:499])

plt.show()


