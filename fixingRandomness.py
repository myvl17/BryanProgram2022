# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:31:33 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn.metrics as skm


def applyAugmentationMethod(df, method, nrows, nvalues, unit=None, noise=None):
    # Reads .txt data frame file
    # df = pd.read_table(df, delimiter=" ", header=None)
    
    # Vector of original and augmented points
    original_points = []
    augmented_points = []
    
    
    if method == "randSwap":
        
        # Creates empty data frame
        augmented_df = pd.DataFrame()
        
        for k in range(0, nrows, 1):
                   
            # Selects random row index
            # random.seed(k)
            random_row = random.randint(0, df.shape[0]-1)
            
            # Adds new row from pre-existing random row
            augmented_df = pd.concat([augmented_df, df.iloc[[random_row]]], ignore_index=True)
            
            
            # Actual Data Augmentation Method:
            # Grabs random row from original data set and appends to new data frame
            # Selects random column from new row and takes random value from same column in original data set
            # Appends random value from original data frame and appends to new row column in new data frame
            for i in range(nvalues):
                
                # Selects random column index
                # random.seed(i)
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
        #print(norm1Distance(original_points, augmented_points))
        
        return finished_df
        
    elif method == "pmOne":
        
        # Reads in the dataset needed, dropping whatever column contains
        # the labels/status

        #df = dftest.drop(columns = dftest.shape[1] - 1)
        
        df1 = df.drop(columns = df.shape[1] - 1)

        # if statement to determine if the number of rows entered is odd
        # The sample function takes random rows from the df
        # in this case it take in the nrows and the # of rows
        
        sample1 = pd.DataFrame()
        sample2 = pd.DataFrame()
        
        if (nrows % 2 == 0):
            '''
            sample1 = df1.sample(n = int(nrows / 2), random_state=(0))
            sample2 = df1.sample(n = int(nrows / 2), random_state=(0))
            '''
            
            for i in range(int(nrows/2)):
                ##random.seed(i)
                sample1 = pd.concat([sample1, df1.iloc[[random.randint(0, df1.shape[0]-1)]]], ignore_index=True)
                sample2 = pd.concat([sample2, df1.iloc[[random.randint(0, df1.shape[0]-1)]]], ignore_index=True)
            
            
        else:
            
            # sample1 = df1.sample(n = int((nrows / 2 ) + 0.5), random_state=(1))
            # sample2 = df1.sample(n = int((nrows / 2) - 0.5), random_state=(1))
            
            
            for k in range(int(nrows / 2 + .5)):
                ##random.seed(k)
                sample1 = pd.concat([sample1, df1.iloc[[random.randint(0, df1.shape[0]-1)]]], ignore_index=True)
                sample2 = pd.concat([sample2, df1.iloc[[random.randint(0, df1.shape[0]-1)]]], ignore_index=True)
            
        # Reset the index in each sample so they increase from 0 to nrows        
        sample1real = sample1.reset_index(drop = True)
        sample2real = sample2.reset_index(drop = True)
        
        
    # Create a list of random numbers
        randomlist = []
        for j in range(0, nvalues):
            ##random.seed(j)
            n = random.randint(0, df.shape[1]-2)
            randomlist.append(n)
            
    # Select one of the random rows then use the random list to 
    # pinpoint one specfic number in the dataframe and add or 
    # subtract the unit specified in the function
        for i in range(len(sample1real)):
            for j in randomlist:
    
                oldValue = sample1real.iloc[i, j]
                newValue = oldValue + unit
                
                # Appends old and new values to augmented_points vector to keep track for distance
                original_points.append(oldValue)
                augmented_points.append(newValue)
                
                # Replace the oldvalue with the new value in the
                # samples set
                sample1real = sample1real.replace(to_replace = oldValue, value = newValue)
                
           
        for i in range(len(sample2real)):
            for j in randomlist:
                oldValue = (sample2real.iloc[i, j])
                newValue = oldValue - unit
                
                # Appends and and new value to augmented_points vector to keep track for distance
                original_points.append(oldValue)
                augmented_points.append(newValue)
                
                
                sample2real = sample2real.replace(to_replace = oldValue, value = newValue)
                
            

        #print(np.linalg.norm(np.array(original_points) - np.array(augmented_points), ord=2)) norm 2
        # Norm 1 distance
        #print(norm1Distance(original_points, augmented_points))
        
        

        # Put the two samples together and mix them
        dfreal = pd.concat([sample1real, sample2real])
        # dfreal = pd.DataFrame(np.random.permutation(dffinaltest))
        
        finished_df = pd.concat([df, dfreal], ignore_index=True)
        
        return finished_df
        
    elif method == "gausNoise":
    #Create a noise matrix
       noise_matrix = pd.DataFrame(np.random.normal(0, noise, size = (nrows, df.shape[1]-1)))
       
       #noise_matrix = pd.DataFrame()
       
       # for k in range(nrows):
       #     #random.seed(k)
       #     noise_matrix = pd.concat([noise_matrix, df.iloc[[random.randint(0, df.shape[1]-1)]]], ignore_index=True)
           
       # print(noise_matrix)
      
      
       if (1 == 0):
           return (df.add(noise_matrix, fill_value = 0))
      
       #add noise to random rows matrix from data set
       else:
           
           # data_portion = df.sample(n = nrows, ignore_index=True)
           
           data_portion = pd.DataFrame()
           for i in range(nrows):
               #random.seed(i)
               data_portion = pd.concat([data_portion, df.iloc[[random.randint(0, df.shape[1]-1)]]], ignore_index=True)
            
           print(data_portion)
           
           added_noise = data_portion.add(noise_matrix, fill_value = None)
           
           
                   
           data_portion.drop(data_portion.columns[-1], axis=1, inplace=True)
           
           finished_df = pd.concat([df, added_noise], ignore_index=True)
           
           for i in range(data_portion.shape[0]):
               for j in range(data_portion.shape[1]):
                   original_points.append(data_portion.loc[i,j])
                   augmented_points.append(added_noise.loc[i,j])

  
           # Norm 1 distance 
           #print(norm1Distance(original_points, augmented_points))
                   
           # print(finished_df)
                   
           return finished_df
    else:
        return None
    
print("HELLO")

# df = pd.read_table('breaking.txt', delimiter=' ', header=None)
# plt.scatter(df[0], df[1], c=df[2])
# plt.show()

# from superFunction import logReg

# df2 = applyAugmentationMethod(df, 'randSwap', 2, 1)
# #df2 = logReg(df2, [0,1], 2, 5)

# plt.scatter(df2[0], df2[1])
# plt.show()

# augment = applyAugmentationMethod(df = 'smallGausDist.txt', method = "gausNoise", nrows = 500, nvalues = 2, noise = 0.05)