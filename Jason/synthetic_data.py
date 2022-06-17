#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:10:07 2022

@author: jasonwhite
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Maintains random values upon each run
np.random.seed(1)

"""

"""

def GaussianDistributions(n_rows, n_columns, median_1, median_2, spread_1, spread_2):
    # Creates first Gaussian distribution
   
    df0 = pd.DataFrame(np.random.normal(median_1, spread_1, size=(int(n_rows/2), n_columns)))
    df1 = pd.DataFrame(np.random.normal(median_1, spread_1, size=(int(n_rows/2), n_columns)))

    df0['label'] = 0
    df1['label'] = 1

    dataset = pd.concat([df0, df1])
    df = pd.DataFrame(np.random.permutation(dataset))




    plt.scatter(df[0], df[1])
    plt.show()

    np.savetxt('Gaussian_Distribution.txt', df)
    
    return df

df = GaussianDistributions(500, 150, 5, 15, 1, 1)
print(df)