# -*x- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:25:33 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = [0.25, 0.5, 0.75, 3.25, 3.75]
y = [0.25, 0.75, 0.5, 2.5, 2.25]
labels = [0,0,0,1,1]

df = pd.DataFrame({'x':x, 'y':y, 'labels':labels})
np.savetxt("breaking.txt", df)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def betterKNN(df, feature_cols, split):
    
    knn = KNeighborsClassifier(n_neighbors=3)
     
    knn.fit(df.iloc[:split, :df.shape[1]-1], df.iloc[:split, df.shape[1]-1])
     
    # Predict on dataset which model has not seen before
    
    #print(accuracy_score(df.iloc[split:, df.shape[1]-1], knn.predict(df.iloc[5:, :2])))
    
    #print(accuracy_score(df.iloc[split:, df.shape[1]-1], knn.predict(df.iloc[5:, :2])))
    
    y_pred = knn.predict(df.iloc[split:, :df.shape[1]-1])
    
    
    return accuracy_score(df.iloc[split:, df.shape[1]-1], y_pred)

    '''
    X = df[feature_cols]
    y = df[df.shape[1]-1]
    
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
                 X, y, train_size=split)
     
    knn = KNeighborsClassifier(n_neighbors=7)
     
    knn.fit(X_train, y_train)
     
    # Predict on dataset which model has not seen before
    return knn.predict(X_test)
    '''

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
def svmClassifier(df, split):
    svc = SVC(gamma=2, C=1, kernel='linear')
    
    svc.fit(df.iloc[:split, :2], df.iloc[:split, 2])
    tmp2 = svc.predict(df.iloc[5:, :2])
    
    for i in range(split, df.shape[0]):
        df.loc[i, df.shape[1]-1] = tmp2[split-i]
        
    return df
    

def betterLogReg(dataset, feature_cols, target, split):
        
    # Feature variables
    X = dataset[feature_cols]
    
    # Target variable
    y = dataset[target]
    
    # Split both x and y into training and testing sets
    
    # Splitting the sets
    # Test_size indicates the ratio of testing to training data ie 0.25:0.75
    # Random_state indicates to select data randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = split, test_size=dataset.shape[0]-split, shuffle = False,  stratify = None) 
    
    
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
        dataset.loc[i, target] = y_pred[i-split]

    
    return dataset


'''
from sklearn.linear_model import LogisticRegression
def betterLogReg(df, split):
    #print(df.iloc[split:, :df.shape[1]-1])
    #print(df.iloc[:split, df.shape[1]-1])
    #print(df.iloc[split:, :df.shape[1]-1])
    
    logreg = LogisticRegression()
    
    logreg.fit(df.iloc[:split, :df.shape[1]-1], df.iloc[:split, df.shape[1]-1])
    y_pred = logreg.predict(df.iloc[split:, :df.shape[1]-1])
    
    for i in range(split, df.shape[0]):
        df.loc[i, df.shape[1]-1] = y_pred[split-i]

    
    return df
'''

from applyAugmentationMethod_distanceMeasuring import applyAugmentationMethod

'''
aug = applyAugmentationMethod(file="breaking.txt", method="pmOne", nrows=5, nvalues=1, unit=0.1)

logReg = betterLogReg(aug, [0,1], 2, 5)

kNNClass = betterKNN(logReg, [0,1], 5)
'''
feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)

aug = applyAugmentationMethod("Generated Gaussian Distribution.txt", "randSwap", 100, 30, noise=0.1)

logReg = betterLogReg(aug, feature_cols, 150, 500)

knnClass = betterKNN(logReg, feature_cols, 500)



















#from applyAugmentationMethod_distanceMeasuring import LogReg

#logReg = LogReg(aug, [0,1], 2, 5)


# predKNN = betterKNN(aug, [0,1], 5)

# acc = betterAcc(logReg, predKNN, 5)

'''
knn = KNeighborsClassifier(n_neighbors=3)
 
knn.fit(df.iloc[:, :2], df.iloc[:, 2])

tmp = knn.predict(aug.iloc[5:, :2])

plt.scatter(aug.iloc[5:, 0], aug.iloc[5:, 1], c=tmp)
'''


# print(skm.accuracy_score(logReg.iloc[5:, logReg.shape[1]-1], predKNN))

'''
from sklearn.svm import SVC
svm =  SVC(gamma=2, C=1, kernel='linear')

svm.fit(df.iloc[:, :2], df.iloc[:, 2])

tmp2 = svm.predict(aug.iloc[5:, :2])

plt.scatter(aug.iloc[5:, 0], aug.iloc[5:, 1], c=tmp2)
'''


plt.scatter(x, y, c=labels)
plt.scatter(aug[0], aug[1], c=aug[2])
plt.yticks(range(0,4,1))
plt.xticks(range(0,4,1))
plt.show()

