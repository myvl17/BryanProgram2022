# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:35:35 2022

@author: jeffr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Maintains random values upon each run
np.random.seed(1)

"""
generatedGaussianDistrubutions Inputs

nrows: Number of rows
ncolumns: Number of columns
median1: First Gaussian distribution median (center)
median2: Second Gaussian distribution median (center)
spread1: First Gaussian distrbiution spread
spread2: Second Gaussian distribution spread

Note:
if label == 0, first Gaussian distribution
if label == 1, second Gaussian distribution
"""

def generateGaussianDistributions(nrows, ncolumns, median1, median2, spread1, spread2):
    # Creates first Gaussian distribution
    label1 = pd.DataFrame(np.random.normal(median1, spread1, size=(int(nrows/2), ncolumns)))
    # Adds new column for label
    label1['label'] = 0
    
    
    # Creates second Gaussian distribution
    label2 = pd.DataFrame(np.random.normal(median2, spread2, size=(int(nrows/2), ncolumns)))
    # Adds new column for label
    label2['label'] = 1
    
    # Combines both Gaussian distributions
    df = pd.concat([label1, label2])
    
    
    # Shuffles Gaussian distributions
    shuffled_df = pd.DataFrame(np.random.permutation(df))
    
    # Creates historgram of Gaussian distributions
    plt.hist(shuffled_df)
    plt.show()
    
    # Saves generated Gaussian distribution in same folder as file
    np.savetxt("Generated Gaussian Distribution.txt", shuffled_df)
    
    return shuffled_df

df = generateGaussianDistributions(500, 150, 5, 15, 1, 1)

