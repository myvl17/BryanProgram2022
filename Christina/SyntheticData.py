# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:17:55 2022

@author: cdiet
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

"""
This function creates uniform synthetic data that exist
within to separate subgroups.

Uniform Synthetic inputs:
    nrows = The number of rows in the data frame you wish to create (even #)
    ncolumns = The number of columns in the data frame you wish to create
    distance = The distance in units between the two groupings of data
    
Uniform Sythetic outputs:
    Saves 3 text files, one of the data, one of the labels, and one of the
    data with the labels. Also should output a scatterplot with two groupings
    that are in a rectangle shape. 

"""

def UniformSythetic(nrows, ncolumns, distance): 
    # Makes sure the random numbers are the same each time the 
    # program runs
    np.random.seed(1)
    
    # The first points will always be between 0 and 1, don't change
    a = 0
    b = 1
    
    # Second set of points is a, either 0 or 1, the distance between points,
    # and 1 added together to create to separate groupings of data
    c = a + distance + 1
    d = b + distance + 1 

    # Create two separate dataframes that fit in two different 
    # ranges on a uniform distribution
    # The number of rows must be divided by two for two even dataframes
    df1 = pd.DataFrame(np.random.uniform(a, b, (int(nrows / 2), ncolumns)))
    df2 = pd.DataFrame(np.random.uniform(c, d, (int(nrows/2), ncolumns)))
    
    # Concatenate the data frames and mix the rows together
    df = pd.concat([df1, df2])
    perm = np.random.permutation(df)
    
    # Turn the permutated data back into a dataframe for use
    dfreal = pd.DataFrame(perm)
    # Save the data frame as a text file if needed
    np.savetxt('synthetic_data.txt', dfreal)
    
    # Creates the labels for the synthetic data by looping through each 
    # value and checking which group it's a part of (could do earlier in code)
    targetvalue = []
    for i in range(len(dfreal)):
        if ((dfreal.iloc[i, 0]) < 1 and (dfreal.iloc[i, 0]) > 0):
            targetvalue.append(0)
        else:
            targetvalue.append(1)
    
    # Add the labels to the synthetic data
    dfreal['status'] = targetvalue
    
    # Save the dataframe to a text file if others want to use
    # One file for the labels, one file for data with labels
    np.savetxt('synthetic_data_labels.txt', targetvalue)
    np.savetxt('synthetic_data_with_labels.txt', dfreal)
    
    # Initiate the plot and graph a scatter of two rows
    # A visual representation of the two groupings
    fig, ax = plt.subplots()
    plt.scatter(df[0], df[1])
    plt.show()

# Testing out the function
print(UniformSythetic(500, 150, 2))