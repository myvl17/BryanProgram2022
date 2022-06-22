# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 13:47:40 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


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
            ##random.seed(k)
            random_row = random.randint(0, df.shape[0]-1)
            
            # Adds new row from pre-existing random row
            augmented_df = pd.concat([augmented_df, df.iloc[[random_row]]], ignore_index=True)
            
            
            # Actual Data Augmentation Method:
            # Grabs random row from original data set and appends to new data frame
            # Selects random column from new row and takes random value from same column in original data set
            # Appends random value from original data frame and appends to new row column in new data frame
            for i in range(nvalues):
                
                # Selects random column index
                ##random.seed(i)
                random_col = random.randint(0, df.shape[1]-2)
                
                # Selects random value from original data frame in the same column
                ##random.seed(i+1)
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
        if (nrows % 2 == 0):
            ##random.seed(1)
            sample1 = df1.sample(n = int(nrows / 2), random_state=(0))
            sample2 = df1.sample(n = int(nrows / 2), random_state=(0))
        else:
            ##random.seed(0)
            sample1 = df1.sample(n = int((nrows / 2 ) + 0.5), random_state=(1))
            sample2 = df1.sample(n = int((nrows / 2) - 0.5), random_state=(1))
            
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
                oldValue = (sample1real.iloc[i, j])
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
        dffinaltest = pd.concat([sample1real, sample2real])
        dfreal = pd.DataFrame(np.random.permutation(dffinaltest))
        
        finished_df = pd.concat([df, dfreal], ignore_index=True)
        
        return finished_df
        
    elif method == "gausNoise":
    #Create a noise matrix
       ##random.seed(1)
       noise_matrix = pd.DataFrame(np.random.normal(0, noise, size = (nrows, df.shape[1]-1)))
       #Add noise to dataset if equal length
      
       
       if len(df) == nrows:
           return (df.add(noise_matrix, fill_value = 0))
      
       #add noise to random rows matrix from data set
       else:
           ##random.seed(0)
           data_portion = df.sample(n = nrows, ignore_index=True)
           
           added_noise = data_portion.add(noise_matrix, fill_value = None)
                   
           data_portion.drop(data_portion.columns[-1], axis=1, inplace=True)
           
           finished_df = pd.concat([df, added_noise], ignore_index=True)
           
           for i in range(data_portion.shape[0]):
               for j in range(data_portion.shape[1]):
                   original_points.append(data_portion.loc[i,j])
                   augmented_points.append(added_noise.loc[i,j])

  
           # Norm 1 distance 
           #print(norm1Distance(original_points, augmented_points))
                   
                   
           return finished_df
    else:
        return None
    
def norm1Distance(original_points, augmented_points):
    return np.mean(np.abs(np.array(original_points) - np.array(augmented_points)))


from sklearn.model_selection import train_test_split
import sklearn.metrics 

def LogReg(dataset, feature_cols, target, split):
        
    # Feature variables
    X = dataset[feature_cols]
    
    # Target variable
    y = dataset[target]
    
    # Split both x and y into training and testing sets
    
    # Splitting the sets
    # Test_size indicates the ratio of testing to training data ie 0.25:0.75
    # Random_state indicates to select data randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = split, shuffle = False,  stratify = None) 
    
    
    # import the class
    from sklearn.linear_model import LogisticRegression
    
    ##random.seed(1)
    
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(max_iter = 10000)
    
    # fit the model with data
    logreg.fit(X_train,y_train)
    
    # create the prediction
    y_pred= logreg.predict(X_test)
    
    
    # Appends predicted labels to NAN
    for i in range(split, dataset.shape[0]):
        dataset.loc[i, target] = y_pred[split-i]
    
    return dataset


from sklearn.neighbors import KNeighborsClassifier

def knnClassifier(dataframe, feature_cols, target):
    X = dataframe[feature_cols]
    y = dataframe[target]
     
    ##random.seed(1)
    knn = KNeighborsClassifier(n_neighbors=7)
     
    knn.fit(X, y)
     
    # Predict on dataset which model has not seen before
    predictions = knn.predict(X)
    
    return predictions


def accuracy(file, predictions):
    # Grabs last column containing labels
    return sklearn.metrics.accuracy_score(file[150], predictions)
    
''' 
# randSwap = applyAugmentationMethod("Generated Gaussian Distribution.txt", "randSwap", 10, 30)
# plt.scatter(pd.read_table(gausDistribution, delimiter=" ", header=None)[0], pd.read_table(gausDistribution, delimiter=" ", header=None)[1], c="b", alpha=0.4)
# plt.scatter(randSwap[0], randSwap[1], c="r", alpha=0.2)
# plt.show()

# pmOne = applyAugmentationMethod("Generated Gaussian Distribution.txt", "pmOne", 100, 30, 0.1)
# plt.scatter(pd.read_table(gausDistribution, delimiter=" ", header=None)[0], pd.read_table(gausDistribution, delimiter=" ", header=None)[1], c="b", alpha=0.4)
# plt.scatter(pmOne[0], pmOne[1], c="r", alpha=0.4)
# plt.show()

# gausNoise = applyAugmentationMethod("Generated Gaussian Distribution.txt", "gausNoise", 1, 30, noise=.05)
# plt.scatter(pd.read_table(gausDistribution, delimiter=" ", header=None)[0], pd.read_table(gausDistribution, delimiter=" ", header=None)[1], c="b", alpha=0.4)
# plt.scatter(gausNoise[0], gausNoise[1], c="r", alpha=0.4)
# plt.show()
'''

'''
pmOne
===================================
Units       Distance       Accuracy
1
.5
.25
0.1
0.05
-----------------------------------

gausNoise
==================================
%noise      Distance      Accuracy
1.00
.75
.50
.30
.10
0.05

----------------------------------

randSwap
==================================
nValues     Distance      Accuracy
100
10
30
50
1

----------------------------------
'''
def distanceAccuracyComparison(dataset, method, nrows, nvalues, feature_cols, target, split, unit=None, noise=None):
    
    ##random.seed(1)
    
    logReg = LogReg(dataset = applyAugmentationMethod(dataset, method, nrows, nvalues, unit, noise), feature_cols = feature_cols, target = target, split = split)
    
    predictions = knnClassifier(logReg, feature_cols, target)
    acc = accuracy(logReg, predictions)
    
    return acc

'''
feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)
    
files = ["Generated Gaussian Distribution.txt", "synthetic_data_with_labels.txt"]

pmOneAcc_Gaus = []
pmOneAcc_Uniform = []
pmOneDist = [0.05, 0.25, 0.5, 0.75, 1]


for j in range(len(pmOneDist)):
    pmOneAcc_Gaus.append(distanceAccuracyComparison(files[0], "pmOne", nrows=100, nvalues=30, unit=pmOneDist[j], feature_cols=feature_cols, target=150, split=500))
    
for j in range(len(pmOneDist)):
    pmOneAcc_Uniform.append(distanceAccuracyComparison(files[0], "pmOne", nrows=100, nvalues=30, unit=pmOneDist[j], feature_cols=feature_cols, target=150, split=500))

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

fig.suptitle("pmOne Augmentation Method")

ax[0].plot(pmOneDist, pmOneAcc_Gaus)
ax[1].plot(pmOneDist, pmOneAcc_Uniform)

ax[0].set_title("Gaussian Distribution")
ax[0].set_ylabel("Accuracy")
ax[0].set_xlabel("Unit")
ax[1].set_title("Uniform Distribution")
ax[1].set_ylabel("Accuracy")
ax[1].set_xlabel("Unit")

ax[0].set_xticks(pmOneDist)

plt.tight_layout()

plt.show()


gausNoiseAcc_Gaus = []
gausNoiseAcc_Uniform = []
gausNoiseDist = [0.05, 0.25, 0.5, 0.75, 1]

for j in range(len(gausNoiseDist)):
    gausNoiseAcc_Gaus.append(distanceAccuracyComparison(files[0], "gausNoise", nrows=100, nvalues=30, noise=gausNoiseDist[j], feature_cols=feature_cols, target=150, split=500))
    
for j in range(len(gausNoiseDist)):
    gausNoiseAcc_Uniform.append(distanceAccuracyComparison(files[0], "gausNoise", nrows=100, nvalues=30, noise=gausNoiseDist[j], feature_cols=feature_cols, target=150, split=500))

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

fig.suptitle("gausNoise Augmentation Method")

ax[0].plot(gausNoiseDist, gausNoiseAcc_Gaus)
ax[1].plot(gausNoiseDist, gausNoiseAcc_Uniform)

ax[0].set_title("Gaussian Distribution")
ax[0].set_ylabel("Accuracy")
ax[0].set_xlabel("Noise %")
ax[1].set_title("Uniform Distribution")
ax[1].set_ylabel("Accuracy")
ax[1].set_xlabel("Noise %")

ax[0].set_xticks(gausNoiseDist)

plt.tight_layout()

plt.show()


randSwapAcc_Gaus = []
randSwapAcc_Uniform = []
randSwapDist = [1, 15, 30, 50, 75, 100]

for j in range(len(randSwapDist)):
    randSwapAcc_Gaus.append(distanceAccuracyComparison(files[0], "randSwap", nrows=100, nvalues=randSwapDist[j], feature_cols=feature_cols, target=150, split=500))
    
for j in range(len(randSwapDist)):
    randSwapAcc_Uniform.append(distanceAccuracyComparison(files[0], "randSwap", nrows=100, nvalues=randSwapDist[j], feature_cols=feature_cols, target=150, split=500))
    
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

fig.suptitle("randSwap Augmentation Method")

ax[0].plot(randSwapDist, randSwapAcc_Gaus)
ax[1].plot(randSwapDist, randSwapAcc_Uniform)

ax[0].set_title("Gaussian Distribution")
ax[0].set_ylabel("Accuracy")
ax[0].set_xlabel("nValues")
ax[1].set_title("Uniform Distribution")
ax[1].set_ylabel("Accuracy")
ax[1].set_xlabel("nValues")

ax[0].set_xticks(randSwapDist)

plt.tight_layout()

plt.show()
'''