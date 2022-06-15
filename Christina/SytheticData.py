# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:17:55 2022

@author: cdiet
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def UniformSythetic(nrows, ncolumns, distance): 
    # Makes sure the random numbers are the same each time the 
    # program runs
    np.random.seed(1)
    
    # The first points will always be between 0 and 1
    a = 0
    b = 1
    
    # Second set of points is a either 0 or 1, the distance between points,
    # and 1 added together
    c = a + distance + 1
    d = b + distance + 1 

    # Create two separate dataframes that fit in two different 
    # ranges on a uniform distribution
    df1 = pd.DataFrame(np.random.uniform(a, b, (int(nrows / 2), ncolumns)))
    df2 = pd.DataFrame(np.random.uniform(c, d, (int(nrows/2), ncolumns)))
    
    # Concatenate the data frames and mix the rows together
    df = pd.concat([df1, df2])
    perm = np.random.permutation(df)
    
    # Turn the permutated data back into a dataframe for use
    dfreal = pd.DataFrame(perm)
    
    targetvalue = []
    for i in range(len(dfreal)):
        if ((dfreal.iloc[i, 0]) < 1 and (dfreal.iloc[i, 0]) > 0):
            targetvalue.append(0)
        else:
            targetvalue.append(1)
    
    # Save the dataframe to a text file if others want to use
    np.savetxt('synthetic_data_labels', targetvalue)
    np.savetxt('synthetic_data', dfreal)
    
    # Initiate the plot and graph a scatter of two rows
    fig, ax = plt.subplots()
    plt.scatter(df[0], df[1])
    plt.show()

print(UniformSythetic(500, 150, 2))