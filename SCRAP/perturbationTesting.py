# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:23:59 2022

@author: jeffr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import random
import sklearn.metrics as skm

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random


from generateRawData import generateRawData
from fixingRandomness import applyAugmentationMethod
from SVC import SVC
from superFunction import logReg
from FunctionReturnsF1 import OkayFunction
from superFunction import superFunction


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def runClassifier(df, classifier, accuracy=None):
    dfdrop = df.drop(columns = df.shape[1] - 1)
    
    results_df = pd.DataFrame(columns = ["Accuracy", "Mean Absolute Error", "Rooted Mean Square Error", "F1 Score"])
    
    if classifier == "kNN":
     
        X = dfdrop
        Y = df[df.shape[1] - 1]
        
        # Split into training and test set
        X_train, X_test, y_train, y_test = train_test_split(
                     X, Y, test_size = 0.2, random_state=0)
         
        knn = KNeighborsClassifier(n_neighbors=2)
         
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
        return f1_accuracy
        #results_df = results_df.append({'F1 Score' : f1_accuracy}, ignore_index=True)

        
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

if 1 == 0:
<<<<<<< Updated upstream
    data = generateRawData(10, 3, -2, 'gaussian')
=======
    data = generateRawData(500, 5, -2, 'gaussian')
>>>>>>> Stashed changes
else:
    None

feature_cols = []
for i in range(0, data.shape[1]-1):
    feature_cols.append(i)
        

<<<<<<< Updated upstream
pmUnit = [0, 0.1, 0.5, 1, 10, 20]
=======
pmUnit = [0.1, 0.5, 1, 10, 20, 100]
>>>>>>> Stashed changes
accuracy = []


for i in range(len(pmUnit)):
<<<<<<< Updated upstream
    augmented = applyAugmentationMethod(df=data, method='pmOne', nrows=500, nvalues=3, unit=pmUnit[i])
=======
    augmented = applyAugmentationMethod(df=data, method='gausNoise', nrows=500, nvalues=pmUnit[i], noise=pmUnit[i])
>>>>>>> Stashed changes
    
    labels = logReg(augmented, feature_cols=feature_cols, target=augmented.shape[1]-1, split=500)
    
    f1 = runClassifier(labels, 'kNN', 'f1')
    
    accuracy.append(f1)
    
print(accuracy)

<<<<<<< Updated upstream
plt.show()
plt.plot(pmUnit, accuracy)
plt.show()
=======
plt.plot(pmUnit, accuracy)
>>>>>>> Stashed changes

