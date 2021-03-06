# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:43:46 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = [.25, .75, .4, .8, .9, 1, 1.1, .95, .925, .85]
y = [.3, .2, .75, .6, .75, .5, .75, .9, 1, 1.1]

labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

plt.scatter(x, y, c=labels)
plt.show()

df = pd.DataFrame({0:x, 1:y, 2:labels})


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
training_x = []
training_y = []
training_labels = []
for i in training:
    training_x.append(x[i])
    training_y.append(y[i])
    training_labels.append(labels[i])

test = [2,3,6,8]
testing_x = []
testing_y = []
testing_labels = []
for i in test:
    testing_x.append(x[i])
    testing_y.append(y[i])
    testing_labels.append(labels[i])


# plt.scatter(training, [.3, .2, .75, .5, .9, 1.1], c=[0,0,0,1,1,1], marker='*', label='training')
# plt.scatter(test, [.75, .6, .75, 1], c=[0,0,1,1], marker='v', label='testing')
plt.scatter(training_x, training_y, c=training_labels, marker='*', label='training', edgecolors='black', s=100)
plt.scatter(testing_x, testing_y, c=testing_labels, marker='D', label='testing', edgecolors='black')
plt.legend()
plt.show()

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
print(accuracy_score(y_test, predicted_values))

from fixingRandomness import applyAugmentationMethod


pmUnits = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 2]
accuracy = []


for i in range(len(pmUnits)):
    plt.clf()
    
    augmented = applyAugmentationMethod(df=df, method='pmOne', nrows=4, nvalues=1, unit=pmUnits[i])
    
    from superFunction import logReg
    from SVC import SVC
    
    labels = logReg(augmented, [0,1], 2, 10)
    
    titleStr = 'pmOne Accuracy with unit of ' + str(pmUnits[i])
    
    plt.scatter(labels.iloc[:10, 0], labels.iloc[:10, 1], c=labels.iloc[:10, 2], marker='o')
    plt.scatter(labels.iloc[10:, 0], labels.iloc[10:, 1], c=labels.iloc[10:, 2], marker='x', s=250)
    plt.xlim(-1, 2)
    plt.ylim(0, 2)
    plt.title(label=titleStr)
    plt.show()
    
    test = [2,3,6,8,9,10,11,12,13]
    
    X_train = labels.iloc[training, :2]
    y_train = labels.iloc[training, 2]
    
    X_test = labels.iloc[test, :2]
    y_test = labels.iloc[test, 2]
    
    
    from sklearn.svm import SVC
    import random
    
    # instantiate the model (using the default parameters)
    #random.seed(1)
    svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)
    
    # fit the model with data
    # svm.fit(X_train,y_train)
    svm.fit(X_train, y_train)
    predicted_values = svm.predict(X_test)
    
     
    # # Predict on dataset which model has not seen before
    # predicted_values = knn.predict(X_test)
        
    (f1_score(y_test, predicted_values))
    accuracy.append(accuracy_score(y_test, predicted_values))    
    
plt.plot(pmUnits, accuracy, marker='o')
plt.title("pmOne Accuracy Experimentation")
plt.xlabel("Unit")
plt.ylabel("Accuracy Percent")
plt.show()

# gausNoise = [0.05, .1, .25, .50, .75, 1]
# accuracy=[]

# for i in range(len(gausNoise)):
#     augmented = applyAugmentationMethod(df=df, method='gausNoise', nrows=4, nvalues=1, noise=gausNoise[i])
    
#     from superFunction import logReg
#     from SVC import SVC
    
#     titleStr = "gausNoise Accuracy with noise of " + str(gausNoise[i] * 100) + "%"
    
#     labels = logReg(augmented, [0,1], 2, 10)
    
#     plt.scatter(labels.iloc[:10, 0], labels.iloc[:10, 1], c=labels.iloc[:10, 2], marker='o')
#     plt.scatter(labels.iloc[10:, 0], labels.iloc[10:, 1], c=labels.iloc[10:, 2], marker='x', s=250)
#     plt.title(label=titleStr)
#     plt.xlim(-1, 3)
#     plt.show()
    
#     test = [2,3,6,8,9,10,11,12,13]
    
#     X_train = labels.iloc[training, :2]
#     y_train = labels.iloc[training, 2]
    
#     X_test = labels.iloc[test, :2]
#     y_test = labels.iloc[test, 2]
    
    
#     # knn = KNeighborsClassifier(n_neighbors=2)
     
#     # knn.fit(X_train, y_train)
    
#     from sklearn.svm import SVC
#     import random
    
#     # instantiate the model (using the default parameters)
#     #random.seed(1)
#     svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)
    
#     # fit the model with data
#     # svm.fit(X_train,y_train)
#     svm.fit(X_train, y_train)
#     predicted_values = svm.predict(X_test)
    
     
#     # # Predict on dataset which model has not seen before
#     # predicted_values = knn.predict(X_test)
        
#     (f1_score(y_test, predicted_values))
#     accuracy.append(accuracy_score(y_test, predicted_values))  
# plt.show()
# plt.plot(gausNoise, accuracy, marker='o')

    

# numSwaps = [0, 1, 2]
# accuracy = []


# for i in range(len(numSwaps)):
#     augmented = applyAugmentationMethod(df=df, method='randSwap', nrows=4, nvalues=numSwaps[i], unit=numSwaps[i])
    
#     from superFunction import logReg
#     from SVC import SVC
    
#     labels = logReg(augmented, [0,1], 2, 10)
    
#     plt.scatter(labels[0], labels[1], c=labels[2])
#     plt.show()
    
#     test = [2,3,6,8,9,10,11,12,13]
    
#     X_train = labels.iloc[training, :2]
#     y_train = labels.iloc[training, 2]
    
#     X_test = labels.iloc[test, :2]
#     y_test = labels.iloc[test, 2]
    
    
#     # knn = KNeighborsClassifier(n_neighbors=2)
     
#     # knn.fit(X_train, y_train)
    
#     from sklearn.svm import SVC
#     import random
    
#     # instantiate the model (using the default parameters)
#     #random.seed(1)
#     svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)
    
#     # fit the model with data
#     # svm.fit(X_train,y_train)
#     svm.fit(X_train, y_train)
#     predicted_values = svm.predict(X_test)
    
     
#     # # Predict on dataset which model has not seen before
#     # predicted_values = knn.predict(X_test)
        
#     (f1_score(y_test, predicted_values))
#     accuracy.append(accuracy_score(y_test, predicted_values))    
    
# plt.show()
# plt.plot(numSwaps, accuracy)  
# plt.show()



    