# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:13:23 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# from generateRawData import generateRawData
# from superFunction import runClassifier


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sklearn.metrics as skm

def runClassifier(df, classifier, accuracy=None):
    
    # dfdrop = df.drop(columns = df.shape[1] - 1)
    
    # results_df = pd.DataFrame(columns = ["Accuracy", "Mean Absolute Error", "Rooted Mean Square Error", "F1 Score"])
    results_df = pd.DataFrame({'Accuracy':[np.NAN], 'Mean Absolute Error':[np.NAN], 'Rooted Mean Square Error':[np.NAN], 'F1 Score':[np.NAN]})
    

    X = df.iloc[:, :df.shape[1]-1]
    Y = df.iloc[:, df.shape[1] - 1]
    
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
                 X, Y, test_size = 0.2, random_state=42)
 
    # random.seed(1)
    svm = SVC(gamma = 2, C = 1, kernel = 'linear', max_iter = 10000, random_state = 0)
    
    # fit the model with data
    # svm.fit(X_train,y_train)
    svm.fit(X_train, y_train)
    predicted_values = svm.predict(X_test)
    
    
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
        
        results_df.iloc[0,0] = acc
        results_df.iloc[0,1] = mae_accuracy
        results_df.iloc[0,2] = rmse_accuracy
        results_df.iloc[0,3] = f1_accuracy
        
    return results_df


'''
numCols = []
# for i in range(1, 500, 10):
#     numCols.append(i)

numCols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
accMes = []


for i in range(len(numCols)):
    df = data.drop(data.columns[numCols[i]:data.shape[1]-1], axis=1)
    acc = runClassifier(df=df, classifier='SVM')

    accMes.append(acc.iloc[0,3])
    
    print(acc)
    

plt.plot(numCols, accMes, marker='o')
plt.show()
'''