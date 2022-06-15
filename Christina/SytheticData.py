# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:17:55 2022

@author: cdiet
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import random


# Makes sure the random numbers are the same each time the 
# program runs
np.random.seed(1)

# Create two separate dataframes that fit in two different 
# ranges on a uniform distribution
df1 = pd.DataFrame(np.random.uniform(0, 1, (250, 150)))
df2 = pd.DataFrame(np.random.uniform(3, 4, (250, 150)))

# Concatenate the data frames and mix the rows together
df = pd.concat([df1, df2])
perm = np.random.permutation(df)

# Turn the permutated data back into a dataframe for use
dfreal = pd.DataFrame(perm)

# Save the dataframe to a text file if others want to use
np.savetxt('test.txt', dfreal)

# Initiate the plot and graph a scatter of two rows
fig, ax = plt.subplots()
plt.scatter(df[0], df[1])
plt.show()
