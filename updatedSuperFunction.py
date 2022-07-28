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


from betterApplyAugmentationMethod import betterApplyAugmentationMethods
    
  
from sklearn.model_selection import train_test_split
def logReg(dataset, split):
    
    # Selects all columns excluding labels
    feature_cols = np.arange(0, dataset.shape[1]-1, 1)
    # Selects only labels column
    target = dataset.shape[1]-1
    
    logDf = dataset.copy(deep=True)
        
   # Feature variables
    X = logDf[feature_cols]
    
    # Target variable
    y = logDf[target]
    
    # Split both x and y into training and testing sets
    
    # Splitting the sets
    # Test_size indicates the ratio of testing to training data ie 0.25:0.75
    # Random_state indicates to select data randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = split, shuffle = False,  stratify = None) 

    # import the class
    from sklearn.linear_model import LogisticRegression
    
    # instantiate the model (using the default parameters)
    # random.seed(1)
    logreg = LogisticRegression(max_iter = 10000)
    
    # fit the model with data
    
    #print(y_train)
    logreg.fit(X_train,y_train)
    
    # create the prediction
    y_pred= logreg.predict(X_test)

    # Appends predicted labels to NAN
    for i in range(split, logDf.shape[0]):
        logDf.iloc[i, target] = y_pred[i - split]
        
    
    return logDf


from pyts.classification import TimeSeriesForest

def ts_dtree(dataset, split):
   
    # Selects all columns except labels
    X = dataset.drop(columns = dataset.shape[1] - 1)
    # Selects only labels column
    y = dataset[dataset.shape[1] - 1]
    
    
    # Splits dataframe into testing and training data, splits between original and augmented
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = split, shuffle = False,  stratify = None) 
    
    # Time Series forest classifier
    clf = TimeSeriesForest(random_state=43)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    for i in range(split, dataset.shape[0]):
        dataset.iloc[i, dataset.shape[1]-1] = y_pred[i - split]
        
    return dataset


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.svm import SVC

def runClassifier(df, classifier, accuracy=None):
    dfdrop = df.drop(columns = df.shape[1] - 1)
    
    # Creates accuracy table dataframe
    results_df = pd.DataFrame(columns = ["Accuracy", "Mean Absolute Error", "Rooted Mean Square Error", "F1 Score"])
    
    if classifier == "kNN":
     
        X = dfdrop
        Y = df[df.shape[1] - 1]
        
        # Split into training and test set
        X_train, X_test, y_train, y_test = train_test_split(
                     X, Y, test_size = 0.2, random_state=42)
         
        knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
         
        knn.fit(X_train, y_train)
         
        # Predict on dataset which model has not seen before
        predicted_values = knn.predict(X_test)
    
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
        Y = df.df[df.shape[1] - 1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size = 0.20, random_state = 0)
        
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        
        predicted_values  =  classifier.predict(X_test)
    
    elif classifier == "ANN":
        
        X = dfdrop
        Y = df.df[df.shape[1] - 1]
        
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
        
    #SVM
    elif classifier == "SVM":
        
        X = dfdrop
        y = df[df.shape[1] - 1]
        
        # Split into training and test set
        X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size = 0.2, random_state=42)
     
        # random.seed(1)
        svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 1000000, random_state = 0)
        
        # fit the model with data
        # svm.fit(X_train,y_train)
        svm.fit(X_train, y_train)
        predicted_values = svm.predict(X_test)
        
    else:
        print("Unknown classifier")
        return None
        

    
    
    #Accuracy
    if (accuracy == "og"): 
        acc = skm.accuracy_score(y_test, predicted_values)
        return acc
        
    elif (accuracy == "mae"):
        mae_accuracy = skm.mean_absolute_error(y_test, predicted_values)
        return mae_accuracy

    
    elif (accuracy == "rmse"):
        rmse_accuracy = skm.mean_squared_error(y_test, predicted_values,
                                                    squared=False)
        return rmse_accuracy

    
    elif(accuracy == "f1"):
        f1_accuracy = skm.f1_score(y_test, predicted_values)
        return f1_accuracy

        
    else:
        acc = skm.accuracy_score(y_test, predicted_values)
        mae_accuracy = skm.mean_absolute_error(y_test, predicted_values)
        rmse_accuracy = skm.mean_squared_error(y_test, predicted_values,
                                                    squared=False)
        f1_accuracy = skm.f1_score(y_test, predicted_values)
        
        # Appends accuracies to accuracy table
        results_df.iloc[0,0] = acc
        results_df.iloc[0,1] = mae_accuracy
        results_df.iloc[0,2] = rmse_accuracy
        results_df.iloc[0,3] = f1_accuracy
        
    return results_df


"""
superFunction applies all methods from the flowchart: augmentation, 
interpretable/uninterpretable classifiers, and accuracy, taking all inputs from
these functions and outputs the accuracy of the augmented data.

Inputs:
    file: A text file containing all raw data with the labels
    method: The augmentation method the user wants to use for the data
    nrows: How many output augmentation rows are wanted
    nvalues: The number of values in each row that need to be augmented
    labels_classifier: interpretable classifier the user wants for labels
    split: The number of rows that contain original data
    classifier: The classifier the user wants to use
    accuracy(optional): Which type of accuracy the user would like to use,
    the default is to output a row of all accuracy measures
    unit(optional): Only for the pmOne augmentation method and is the unit the 
    augmented data will differ from original data by
    noise(optional): Only for the gausNoise augmentation method and denotes the
    percent by which the augmented data varies from original data
    
    
Outputs:
    Gives a row of accuracy measures or the accuracy measure chosen by the user
"""
def superFunction(file, method, nrows, nvalues, labels_classifier, split, classifier, accuracy=None, unit=None, noise=None):
    df = pd.read_table(file, delimiter=" ", header=None)

    augmentation = betterApplyAugmentationMethods(df, method, nrows, nvalues, unit=unit, noise=noise)
    
    
    if str(labels_classifier).lower == 'logreg' or 'log regression' or 'logistic regression':
        logRegression = logReg(augmentation, split)
        
    elif str(labels_classifier).lower == 'decisiontree' or 'decision tree':
        ts_dtree(augmentation, split)
    else:
        print('Unknown classifier')
        return None
    

    classifier = runClassifier(logRegression, classifier, accuracy)
    
    return classifier
    




