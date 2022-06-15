# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:41:43 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


df = pd.read_table("Gaussian Distribution Data Set.txt", delimiter=" ", header=None)

def rand_value_col(perturbation):
    
    # Copies original data set
    augmented_df = df.copy(deep=True)
    
    for k in range(0, perturbation, 1):
               
        # Selects random row index
        random_row = random.randint(0, augmented_df.shape[0]-1)
        
        # Adds new row from pre-existing random row
        augmented_df = pd.concat([augmented_df, augmented_df.iloc[[random.randint(0, augmented_df.shape[0]-1)]]], ignore_index=True)
        
        
        # Performs 30 pertubations
        for i in range(150):
            
            # Selects random column index
            random_col = random.randint(0, augmented_df.shape[1]-1)
            
            # Selects random value from row and column
            rand_value = augmented_df.iloc[random_row][random_col]
            
            # Selects random index location and changes value
            augmented_df.iloc[-1][i] = rand_value # THIS BREAKS EVERYTHING
    
    return augmented_df

 
test = rand_value_col(100)

fig, ax = plt.subplots(1,2, sharey=True) #figsize=(50,20)

ax[0].scatter(df[1], df[2], alpha=0.3)

ax[1].scatter(test[1], test[2], alpha=0.3)

# ax[0].hist(df, density=True, bins=50)
# ax[1].hist(test, density=True, bins=50)
plt.show()

# print(df.describe(include='all'))
# print(test.describe(include='all'))



