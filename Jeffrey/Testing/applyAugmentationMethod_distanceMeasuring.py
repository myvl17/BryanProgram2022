# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 13:47:40 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

gausDistribution = "Generated Gaussian Distribution.txt"

def applyAugmentationMethod(file, method, nrows, nvalues, unit=None, noise=None):
    # Reads .txt data frame file
    df = pd.read_table(file, delimiter=" ", header=None)
    
    # Vector of original and augmented points
    original_points = []
    augmented_points = []
    
    
    if method == "randSwap":
        
        # Creates empty data frame
        augmented_df = pd.DataFrame()
        
        for k in range(0, nrows, 1):
                   
            # Selects random row index
            random_row = random.randint(0, df.shape[0]-1)
            
            # Adds new row from pre-existing random row
            augmented_df = pd.concat([augmented_df, df.iloc[[random_row]]], ignore_index=True)
            
            
            # Actual Data Augmentation Method:
            # Grabs random row from original data set and appends to new data frame
            # Selects random column from new row and takes random value from same column in original data set
            # Appends random value from original data frame and appends to new row column in new data frame
            for i in range(nvalues):
                
                # Selects random column index
                random_col = random.randint(0, df.shape[1]-2)
                
                # Selects random value from original data frame in the same column
                rand_value = df.iloc[random.randint(0, df.shape[0]-1)][random_col]
                
                # Appends original and old value to keep track of distances
                original_points = augmented_df.iloc[-1][random_col]
                augmented_points.append(rand_value)
                
                # Appends rand_value to new column
                augmented_df.iloc[-1][random_col] = rand_value
                
                
                
                
        # Removes label column
        augmented_df.drop(df.columns[-1], axis=1, inplace=True)
        
        finished_df = pd.concat([df, augmented_df], ignore_index=True)
        
        # Norm 1 distance 
        print(np.mean(np.abs(np.array(original_points) - np.array(augmented_points))))
        
        return finished_df
        
    elif method == "pmOne":
        # Reads in the dataset needed, dropping whatever column contains
        # the labels/status

        #df = dftest.drop(columns = dftest.shape[1] - 1)
        
        df1 = df.drop(columns = df.shape[1] - 1)

        # if statement to determine if the number of rows entered is odd
        # The sample function takes random rows from the df
        # in this case it take in the nrows and the # of rows
        if (nrows % 2 == 0):
            sample1 = df1.sample(n = int(nrows / 2))
            sample2 = df1.sample(n = int(nrows / 2))
        else:
            sample1 = df1.sample(n = int((nrows / 2 ) + 0.5))
            sample2 = df1.sample(n = int((nrows / 2) - 0.5))
            
        # Reset the index in each sample so they increase from 0 to nrows        
        sample1real = sample1.reset_index(drop = True)
        sample2real = sample2.reset_index(drop = True)
        
    # Create a list of random numbers
        randomlist = []
        for j in range(0, nvalues):
            n = random.randint(0, 149)
            randomlist.append(n)
            
    # Select one of the random rows then use the random list to 
    # pinpoint one specfic number in the dataframe and add or 
    # subtract the unit specified in the function
        for i in range(len(sample1real)):
            for j in randomlist:
                oldValue = (sample1real.iloc[i, j])
                newValue = oldValue + unit
                
                # Appends old and new values to augmented_points vector to keep track for distance
                original_points.append(oldValue)
                augmented_points.append(newValue)
                
                # Replace the oldvalue with the new value in the
                # samples set
                sample1real.replace(to_replace = oldValue, value = newValue)
                
           
        for i in range(len(sample2real)):
            for j in randomlist:
                oldValue = (sample2real.iloc[i, j])
                newValue = oldValue - unit
                
                # Appends and and new value to augmented_points vector to keep track for distance
                original_points.append(oldValue)
                augmented_points.append(newValue)
                
                
                sample2real.replace(to_replace = oldValue, value = newValue)
                
            

        #print(np.linalg.norm(np.array(original_points) - np.array(augmented_points), ord=2)) norm 2
        # Norm 1 distance
        print(np.mean(np.abs(np.array(original_points) - np.array(augmented_points))))
        
        

        # Put the two samples together and mix them
        dffinaltest = pd.concat([sample1real, sample2real])
        dfreal = pd.DataFrame(np.random.permutation(dffinaltest))
        
        finished_df = pd.concat([df, dfreal], ignore_index=True)
        
        return finished_df
        
    elif method == "gausNoise":
    #Create a noise matrix
       noise_matrix = pd.DataFrame(np.random.normal(0, noise, size = (nrows, 150)))
       #Add noise to dataset if equal length
      
       
       if len(df) == nrows:
           return (df.add(noise_matrix, fill_value = 0))
      
       #add noise to random rows matrix from data set
       else:
           data_portion = df.sample(n = nrows, ignore_index=True)
           
           added_noise = data_portion.add(noise_matrix, fill_value = 0)
                   
           data_portion.drop(data_portion.columns[-1], axis=1, inplace=True)
           
           finished_df = pd.concat([df, added_noise], ignore_index=True)
           
           for i in range(data_portion.shape[0]):
               for j in range(data_portion.shape[1]):
                   original_points.append(data_portion.loc[i,j])
                   augmented_points.append(added_noise.loc[i,j])

  
           # Norm 1 distance 
           print(np.mean(np.abs(np.array(original_points) - np.array(augmented_points))))
                   
                   
           return finished_df
    else:
        return None
    
    
randSwap = applyAugmentationMethod("Generated Gaussian Distribution.txt", "randSwap", 100, 30)
plt.scatter(pd.read_table(gausDistribution, delimiter=" ", header=None)[0], pd.read_table(gausDistribution, delimiter=" ", header=None)[1], c="b", alpha=0.4)
plt.scatter(randSwap[0], randSwap[1], c="r", alpha=0.2)
plt.show()

# pmOne = applyAugmentationMethod("Generated Gaussian Distribution.txt", "pmOne", 100, 30, 0.1)
# plt.scatter(pd.read_table(gausDistribution, delimiter=" ", header=None)[0], pd.read_table(gausDistribution, delimiter=" ", header=None)[1], c="b", alpha=0.4)
# plt.scatter(pmOne[0], pmOne[1], c="r", alpha=0.4)
# plt.show()

# gausNoise = applyAugmentationMethod("Generated Gaussian Distribution.txt", "gausNoise", 1, 30, noise=.05)
# plt.scatter(pd.read_table(gausDistribution, delimiter=" ", header=None)[0], pd.read_table(gausDistribution, delimiter=" ", header=None)[1], c="b", alpha=0.4)
# plt.scatter(gausNoise[0], gausNoise[1], c="r", alpha=0.4)
# plt.show()


