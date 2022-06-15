# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:48:14 2022

@author: jeffr
"""

import pandas as pd
import matplotlib.pyplot as plt
import random

"""
randSwap Inputs

file: file name, NOTE: DELIMETER IS SET TO SPACES AND THERE IS NO HEADER
perturbations: number of new rows to generate
ncols: number of columns to substitute with random value
"""

def randSwap(file, perturbations, ncols):
    # Reads .txt data frame file
    df = pd.read_table(file, delimiter=" ", header=None)
    
    # Creates empty data frame
    augmented_df = pd.DataFrame()
    
    for k in range(0, perturbations, 1):
               
        # Selects random row index
        random_row = random.randint(0, df.shape[0]-1)
        
        # Adds new row from pre-existing random row
        augmented_df = pd.concat([augmented_df, df.iloc[[random_row]]], ignore_index=True)
        
        
        # Actual Data Augmentation Method:
        # Grabs random row from original data set and appends to new data frame
        # Selects random column from new row and takes random value from same column in original data set
        # Appends random value from original data frame and appends to new row column in new data frame
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


"""
Graph to visualize augmented data frame versus original

"""
fix, ax = plt.subplots(1,2, sharey=True)

ax[0].hist(augmented)
ax[0].set_title("Augmented")


ax[1].hist(pd.read_table("Generated Gaussian Distribution.txt", delimiter=" ", header=None))
ax[1].set_title("Original")
plt.show()