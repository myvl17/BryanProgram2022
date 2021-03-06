# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:33:06 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn.metrics as skm



def generateRawData(nrows, ncolumns, distance, distribution, name):
    if distribution == "uniform":
        
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
        df1['labels'] = 0
        df2 = pd.DataFrame(np.random.uniform(c, d, (int(nrows/2), ncolumns)))
        df2['labels'] = 1
        
        # Concatenate the data frames and mix the rows together
        df = pd.concat([df1, df2])
        perm = np.random.permutation(df)
        
        # Turn the permutated data back into a dataframe for use
        dfreal = pd.DataFrame(perm)
        
        # Save the dataframe to a text file if others want to use
        
        fig, ax = plt.subplots()
        plt.scatter(df[0], df[1], c=df[df.shape[1]-1])
        plt.title("Random Uniform Distribution")
        plt.show()
        
        return dfreal
    else:
        # Creates first Gaussian distribution
        label1 = pd.DataFrame(np.random.normal(-5, (10 - distance)/(2), size=(int(nrows/2), ncolumns)))
        # Adds new column for label
        label1['label'] = 0
        
        
        # Creates second Gaussian distribution
        label2 = pd.DataFrame(np.random.normal(5, (10 - distance)/(2), size=(int(nrows/2), ncolumns)))
        # Adds new column for label
        label2['label'] = 1
        
        # Combines both Gaussian distributions
        df = pd.concat([label1, label2])
        
        
        # Shuffles Gaussian distributions
        shuffled_df = pd.DataFrame(np.random.permutation(df))
        
        # Creates historgram of Gaussian distributions
        plt.scatter(shuffled_df[0], shuffled_df[1], c=shuffled_df[shuffled_df.shape[1]-1])
        plt.title("Random Gaussian Distribution")
        plt.show()
        
        # Saves generated Gaussian distribution in same folder as file
        
        return shuffled_df


def applyAugmentationMethod(df, method, nrows, nvalues, unit=None, noise=None):
    # Reads .txt data frame file
    # df = pd.read_table(file, delimiter=" ", header=None)
    
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
        ##random.seed(1)
        
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
            ##random.seed(1)
            '''
            sample1 = df1.sample(n = int(nrows / 2), random_state=(0))
            sample2 = df1.sample(n = int(nrows / 2), random_state=(0))
            '''
            
            for i in range(int(nrows/2)):
                ##random.seed(i)
                sample1 = pd.concat([sample1, df1.iloc[[random.randint(0, df1.shape[0]-1)]]], ignore_index=True)
                sample2 = pd.concat([sample2, df1.iloc[[random.randint(0, df1.shape[0]-1)]]], ignore_index=True)
            
            
        else:
            ##random.seed(0)
            
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
    
    
  
from sklearn.model_selection import train_test_split
def logReg(dataset, feature_cols, target, split):
        
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
    
    # instantiate the model (using the default parameters)
    ##random.seed(1)
    logreg = LogisticRegression(max_iter = 10000)
    
    # fit the model with data
    
    #print(y_train)
    logreg.fit(X_train,y_train)
    
    # create the prediction
    y_pred= logreg.predict(X_test)

    # Appends predicted labels to NAN
    for i in range(split, dataset.shape[0]):
        dataset.loc[i, target] = y_pred[i - split]
    
    return dataset


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from sklearn import svm

def runClassifier(df, classifier, accuracy=None):
    dfdrop = df.drop(columns = df.shape[1] - 1)
    
    results_df = pd.DataFrame(columns = ["Accuracy", "Mean Absolute Error", "Rooted Mean Square Error", "F1 Score"])
    
    if classifier == "kNN":
     
        X = dfdrop
        Y = df[df.shape[1] - 1]
        
        # Split into training and test set
        X_train, X_test, y_train, y_test = train_test_split(
                     X, Y, test_size = 0.1, random_state=42)
         
        knn = KNeighborsClassifier(n_neighbors=32, weights = 'distance')
         
        knn = knn.fit(X_train, y_train)
         
        # Predict on dataset which model has not seen before
        predicted_values = knn.predict(X_test)
        
        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        print(clf.score(X_test, y_test))
    
    elif classifier == "D_tree":
        
        X = dfdrop
        Y = df.df[df.shape[1] - 1]
        
        X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100)
        
        clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
        
        clf_gini.fit(X_train, y_train)
        
        predicted_values = clf_gini.predict(X_test)
        
    elif classifier == "K_cluster":
        
        x = df.iloc[:,1:len(df.columns) - 1] 

        kmeans = KMeans(2)
        kmeans.fit(x)

        predicted_values = kmeans.fit_predict(x)

        
    elif classifier == "Naive_bayes":
        
        X = dfdrop
        Y = df[df.shape[1] - 1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size = 0.20, random_state = 0)
        
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        
        predicted_values  =  classifier.predict(X_test)
    
    elif classifier == "ANN":
        
        X = dfdrop
        Y = df[df.shape[1] - 1]
        
       #Splitting dataset into training and testing dataset
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=5,random_state=42, shuffle= False)

        #Performing Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        #Initialising Artificial Neural Network
        ann = tf.keras.models.Sequential()

        #Adding Hidden Layers
        ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
        ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
        
        #Adding output layers
        ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

        #compiling the Artificial Neural Network
        ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

        #Fitting the Artificial Neural Network
        ann.fit(X_train,Y_train,batch_size=32,epochs = 100)

        #Generate the predicted labels
        first_predicted_values = ann.predict(X_test)
        second_predicted_labels = first_predicted_values > .5
        final_predicted_labels  = second_predicted_labels* 1
        predicted_values = final_predicted_labels
    
    #Accuracy
    if (accuracy == "og"): 
        acc = skm.accuracy_score(y_test, predicted_values)
        results_df = results_df.append({'Accuracy' : acc}, ignore_index=True)
        
    elif (accuracy == "mae"):
        mae_accuracy = skm.mean_absolute_error(y_test, predicted_values)
        results_df = results_df.append({'Mean Absolute Error' : mae_accuracy}, ignore_index=True)

    
    elif (accuracy == "rmse"):
        rmse_accuracy = skm.mean_squared_error(y_test, predicted_values,
                                                    squared=False)
        results_df = results_df.append({'Rooted Mean Square Error' : rmse_accuracy}, ignore_index=True)

    
    elif(accuracy == "f1"):
        f1_accuracy = skm.f1_score(y_test, predicted_values)
        results_df = results_df.append({'F1 Score' : f1_accuracy}, ignore_index=True)

        
    else:
        acc = skm.accuracy_score(y_test, predicted_values)
        mae_accuracy = skm.mean_absolute_error(y_test, predicted_values)
        rmse_accuracy = skm.mean_squared_error(y_test, predicted_values,
                                                    squared=False)
        f1_accuracy = skm.f1_score(y_test, predicted_values)
        
        results_df = results_df.append({'Accuracy' : acc, 
                           'Mean Absolute Error':mae_accuracy,
                           'Rooted Mean Square Error':rmse_accuracy,
                           'F1 Score':f1_accuracy}, ignore_index=True)
        
        
    return results_df

def superFunction(file, method, nrows, nvalues, feature_cols, target, split, classifier, accuracy=None, unit=None, noise=None):
    augmentation = applyAugmentationMethod(file, method, nrows, nvalues, unit=unit, noise=noise)
    
    logRegression = logReg(augmentation, feature_cols, target, split)
    classifier = runClassifier(logRegression, classifier)
    
    print(classifier)
    
  

# data = generateRawData(16, 2, -3, 'gaussian', 'test')
# print(runClassifier(data, 'kNN'))

# save = np.savetxt('small_df.txt', data)

# feature_cols = []
# for i in range(0, 149, 1):
#     feature_cols.append(i)
    
# print(runClassifier(df, 'kNN'))
    
# plsBreak = superFunction(file=df, method='pmOne', nrows=10, nvalues=1, feature_cols=[0,1], target=2, split=16, unit=0.1, classifier='kNN')
    
# test = superFunction(file='Generated Gaussian Distribution.txt', method='pmOne', nrows=500, nvalues=150, unit=0.1, feature_cols=feature_cols, target=150, split=500, classifier='kNN')


# test = superFunction(file='Generated Gaussian Distribution.txt', method='randSwap', nrows=100, nvalues=50, noise=0.1, feature_cols=feature_cols, target=150, split=500, classifier='kNN')

# df = pd.read_table('breaking.txt', delimiter=" ", header=None)
# test = applyAugmentationMethod(df, 'pmOne', 10, 1, unit=0.1)

# feature_cols = []
# for i in range(0, 150, 1):
#     feature_cols.append(i)
    
# df2 = pd.read_table('Gaussian_Data_-5_Unit.txt', delimiter=' ', header=None)
# test4 = runClassifier(df2, 'kNN')
# print(test4)

# plt.scatter(df2[0], df2[1], c=df2[150], alpha=0.5)
# plt.show()

# # test = superFunction(file=df2, method='randSwap', nrows=400, nvalues=30, feature_cols=feature_cols, target=150, split=500, classifier='kNN')
# test = applyAugmentationMethod(df=df2, method='pmOne', unit=10, nrows=200, nvalues=75)
# test2 = logReg(test, feature_cols, 150, 500)

# plt.scatter(test2[0], test2[1], c=test2[150], alpha=0.4)
# plt.show()

# test3 = runClassifier(test2, 'kNN')
# print(test3)
