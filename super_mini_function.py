#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:42:02 2022

@author: jasonwhite
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import random
import sklearn.metrics as skm


# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
#import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def OkayFunction(data, classifier, accuracy=None):
    df = pd.read_table(data, delimiter = " ", header = None) 
    dfdrop = df.drop(columns = df.shape[1] - 1)
    
    results_df = pd.DataFrame(columns = ["Accuracy", "Mean Absolute Error", "Rooted Mean Square Error", "F1 Score"])
    
    if classifier == "kNN":
     
        X = dfdrop
        Y = df[df.shape[1] - 1]
        
        # Split into training and test set
        X_train, X_test, y_train, y_test = train_test_split(
                     X, Y, test_size = 0.2, random_state=42)
         
        knn = KNeighborsClassifier(n_neighbors=7)
         
        knn.fit(X_train, y_train)
         
        # Predict on dataset which model has not seen before
        predicted_values = knn.predict(X_test)
    
    elif classifier == "D_tree":
        
        X = dfdrop
        Y = df[df.shape[1] - 1]
        
        X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100)
        
        clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
        
        clf_gini.fit(X_train, y_train)
        
        predicted_values = clf_gini.predict(X_test)
        
    elif classifier == "K_cluster":
        
        x = df.iloc[:,1:len(df.columns) - 1] 
        y = df[df.shape[1] - 1]
        X_train, X_test, y_train, y_test = train_test_split(
              x, y, test_size = 0.2, random_state=42)

        kmeans = KMeans(2)
        kmeans.fit(X_train, y_train)

        predicted_values = kmeans.fit_predict(X_test)

        
    elif classifier == "Naive_bayes":
        
        X = dfdrop
        Y = df[df.shape[1] - 1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size = 0.20, random_state = 0)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        
        predicted_values  =  classifier.predict(X_test)
    
    elif classifier == "ANN":
        print("Hello")
        # X = dfdrop
        # Y = df[df.shape[1] - 1]
        
        # X_train,X_test,Y_train,Y_test = train_test_split(
        #     X,Y,test_size=0.2,random_state=0)
        
        # sc = StandardScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)
        
        # ann = tf.keras.models.Sequential()
        
        # ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
        # ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
        # ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
        
        # ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
        # ann.fit(X_train,Y_train,batch_size=32,epochs = 100)
        
        # predicted_values = ann.predict(X_test)
        
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
    
    
df = OkayFunction("Gaussian_Distribution.txt", classifier= "Naive_bayes")
print(df)