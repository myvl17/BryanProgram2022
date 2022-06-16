# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:24:23 2022

@author: cdiet
"""
# Create uniformly distributed data and augment

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import random

"""
RandUnit inputs:
    dataset = The name of the file that contains the synthetic data with 
    the labels.
    numbRows = The number of rows that you want to augment.
    unit =  The amount you want to add and subtract randomly from the rows
    labels =  The number of the column that contains the labels/status

RandUnit outputs: 
    Outputs 3 text files, the augmented data, the augmented data with the 
    original data, and the original data, augmented data, and labels (contains
    Nan for the augmented labels)
"""

# Create the function that utilizes the sythetic data
def RandUnit(dataset, numbRows, unit, labels):
    
    # Reads in the dataset needed, dropping whatever column contains
    # the labels/status
    dftest = pd.read_table(dataset, delimiter = " ", header = None) 
    df = dftest.drop(columns = labels)

    # if statement to determine if the number of rows entered is odd
    # The sample function takes random rows from the df
    # in this case it take in the NumbRows and the # of rows
    if (numbRows % 2 == 0):
        sample1 = df.sample(n = int(numbRows / 2))
        sample2 = df.sample(n = int(numbRows / 2))
    else:
        sample1 = df.sample(n = int((numbRows / 2 ) + 0.5))
        sample2 = df.sample(n = int((numbRows / 2) - 0.5))
        
    # Reset the index in each sample so they increase from 0 to NumbRows        
    sample1real = sample1.reset_index(drop = True)
    sample2real = sample2.reset_index(drop = True)
    
# Create a list of random numbers
    randomlist = []
    for j in range(0, numbRows):
        n = random.randint(0, 149)
        randomlist.append(n)
        
# Select one of the random rows then use the random list to 
# pinpoint one specfic number in the dataframe and add or 
# subtract the unit specified in the function
    for i in range(len(sample1real)):
        for j in randomlist:
            oldValue = (sample1real.iloc[i, j])
            newValue = oldValue + unit
            # Replace the oldvalue with the new value in the
            # samples set
            sample1real.replace(to_replace = oldValue, value = newValue)
       
    for i in range(len(sample2real)):
        for j in randomlist:
            oldValue = (sample2real.iloc[i, j])
            newValue = oldValue - unit
            sample2real.replace(to_replace = oldValue, value = newValue)
    

    # Put the two samples together and mix them
    dffinaltest = pd.concat([sample1real, sample2real])
    perm = np.random.permutation(dffinaltest)
    
    # Save back to a dataframe to concatenate with original data
    dfreal = pd.DataFrame(perm)

    # Put the two samples with the original synthetic data
    dffinaltest2 = pd.concat([df, dfreal])

    # Reset the index again so it increases from 0 to n
    dffinal = dfreal.reset_index(drop = True)
    
    dffinal2 = dffinaltest2.reset_index(drop = True)

    # Add too the original scatterplot with a different
    #alpha to show the new points
    plt.scatter(dffinal[0], dffinal[1], alpha = 0.5) 
    plt.show()
    
    # Save dataframe as a text file to be used outside
    # of this function, one with just augmented and one 
    # with original and augmented
    np.savetxt('augmented_data.txt', dffinal)
    np.savetxt('augmented_original.txt', dffinal2)
    
    # Read back in the synthetic data labels to create a text file with
    # original, sythetic data, and labels
    name = pd.read_table('synthetic_data_labels.txt', delimiter = " ", header = None)
    dffinal2['status'] = name
    
    # Save the text file
    np.savetxt('augmented_original_label.txt', dffinal2)
    