# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:48:14 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

def randSwap(file, perturbations, ncols):
    # Reads .txt data frame file
    df = pd.read_table(file, delimiter=" ", header=None)
    
    # Renames label column
    df.rename(columns = {150: 'label'}, inplace = True)
    
    # Copies original data set
    augmented_df = pd.DataFrame()
    
    for k in range(0, perturbations, 1):
               
        # Selects random row index
        random_row = random.randint(0, df.shape[0]-1)
        
        # Adds new row from pre-existing random row
        augmented_df = pd.concat([augmented_df, df.iloc[[random_row]]], ignore_index=True)
        
        
        # Changes ncols amount of rows
        for i in range(ncols):
            
            # Selects random column index
            random_col = random.randint(0, df.shape[1]-2)
            
            # Selects random value from original data frame in the same column
            rand_value = df.iloc[random.randint(0, df.shape[0]-1)][random_col] # BREAKS THINGS
            
            # Appends rand_value to new column
            augmented_df.iloc[-1][random_col] = rand_value
            
    # Removes label column
    augmented_df.drop(df.columns[-1], axis=1, inplace=True)
    return augmented_df

augmented = randSwap("Generated Gaussian Distribution.txt", 1000, 30)

fix, ax = plt.subplots(1,2, sharey=True)


ax[0].hist(augmented)
ax[0].set_title("Augmented")


ax[1].hist(pd.read_table("Generated Gaussian Distribution.txt", delimiter=" ", header=None))
ax[1].set_title("Original")
plt.show()