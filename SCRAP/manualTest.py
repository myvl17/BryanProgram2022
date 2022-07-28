# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:43:46 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


distance = [-1.0, -0.5, -0.1, -0.12, -0.05, -0.01, 0.05, 0.1, 0.5, 1.0]

rawAcc = []
augmentAcc = []

for dist in distance:
    x = [.25, .75, .4, .8, .9, 1, 1.1, .95, .925, .85]
    y = [.3, .2, .75, .6, .75, .5, .75, .9, 1, 1.1]

    i = 0
    x1 = x[:5]
    for value in x1:
        value1 = value - (dist)
        x[i] = value1
        i+=1
        
    x2 = x[5:]
    for value in x2:
        value1 = value  + (dist)
        x[i] = value1
        i+=1
        
    
    i = 0
    y1 = y[:5]
    for value in y1:
        value1 = value - (dist)
        y[i] = value1
        i+=1
     
    y2 = y[5:]
    for value in y2:
        value1 = value + (dist)
        y[i] = value1
        i+=1

    
    # print(x)
    # print(y)
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    
    plt.scatter(x, y, c=labels)
    
    df = pd.DataFrame({0:x, 1:y, 2:labels})
    
    
    print(df)
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.cluster import KMeans
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    
    import tensorflow as tf
    
    '''
    0
    1
    4
    
    5
    7
    9
    '''
        
    training = [0,1,4,5,7,9]
    test = [2,3,6,8]
    
    X_train = df.iloc[training, :2]
    y_train = df.iloc[training, 2]
    
    X_test = df.iloc[test, :2]
    y_test = df.iloc[test, 2]
    
    knn = KNeighborsClassifier(n_neighbors=2)
     
    knn.fit(X_train, y_train)
     
    # Predict on dataset which model has not seen before
    predicted_values = knn.predict(X_test)
    
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    
    (f1_score(y_test, predicted_values))
    rawAcc.append(accuracy_score(y_test, predicted_values))
    print(accuracy_score(y_test, predicted_values))
    
    from fixingRandomness import applyAugmentationMethod
    
    augmented = applyAugmentationMethod(df=df, method='pmOne', nrows=4, nvalues=1, unit = 0.1)
    
    from superFunction import logReg
    from SVC import SVC
    
    labels = logReg(augmented, [0,1], 2, 10)
    print(labels)
    
    test = [2,3,6,8,9,10,11,12,13]
    
    X_train = labels.iloc[training, :2]
    y_train = labels.iloc[training, 2]
    
    X_test = labels.iloc[test, :2]
    y_test = labels.iloc[test, 2]
    
    
    # knn = KNeighborsClassifier(n_neighbors=2)
     
    # knn.fit(X_train, y_train)
    
    from sklearn.svm import SVC
    import random
    
    # instantiate the model (using the default parameters)
    random.seed(1)
    svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)
    
    # fit the model with data
    # svm.fit(X_train,y_train)
    svm.fit(X_train, y_train)
    predicted_values = svm.predict(X_test)
    
     
    # # Predict on dataset which model has not seen before
    # predicted_values = knn.predict(X_test)
    
    
    
    
    
    (f1_score(y_test, predicted_values))
    augmentAcc.append(accuracy_score(y_test, predicted_values))
    print(accuracy_score(y_test, predicted_values))    
        
    plt.scatter(labels[0], labels[1], c=labels[2])
    plt.show()
    
    plt.scatter(X_test[0], X_test[1], c=predicted_values)
    plt.show()
        
        
    # +(np.random.normal(size = X_test[0].shape)) * 0.05

fig, ax = plt.subplots(1, 2)

ax[0].plot(distance, rawAcc, marker = 'd', mfc = 'red')
ax[0].set_title("Raw Data Accuracy")
ax[0].set_ylabel("Accuracy")
ax[0].set_xlabel("Distance")

ax[1].plot(distance, augmentAcc, marker = 'd', mfc='red')
ax[1].set_title("Augmented Data Accuracy")
ax[1].set_xlabel("Distance")

plt.tight_layout()

        
        