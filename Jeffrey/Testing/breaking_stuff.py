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
    
    y_pred = knn.predict(df.iloc[split:, :df.shape[1]-1])
    
    #print(f1_score(df.iloc[split:, df.shape[1]-1], y_pred))
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
from sklearn.metrics import accuracy_score, f1_score
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


def superFunction(file, method, nrows, nvalues, feature_cols, target, split, unit=None, noise=None):
    aug = applyAugmentationMethod(file, method, nrows, nvalues, unit, noise)
    
    logReg = betterLogReg(aug, feature_cols, target, split)
    
    # plt.scatter(logReg[0], logReg[1], c=logReg[logReg.shape[1]-1])
    # plt.show()
    
    knnClass = betterKNN(logReg, feature_cols, split)
    
    return knnClass


feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)
    
# print(superFunction(file="Generated Gaussian Distribution.txt", method="randSwap", nrows=500, nvalues=100, unit=10, feature_cols=feature_cols, target=150, split=500))

files = ["Generated Gaussian Distribution.txt", "synthetic_data_with_labels.txt"]

pmOneAcc_Gaus = []
pmOneAcc_Uniform = []
pmOneDist = [0.1, 0.5, 0.5, 0.75, 1]


for j in range(len(pmOneDist)):
    pmOneAcc_Gaus.append(superFunction(files[0], "pmOne", nrows=100, nvalues=30, unit=pmOneDist[j], feature_cols=feature_cols, target=150, split=500))
    
for j in range(len(pmOneDist)):
    pmOneAcc_Uniform.append(superFunction(files[0], "pmOne", nrows=100, nvalues=30, unit=pmOneDist[j], feature_cols=feature_cols, target=150, split=500))

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
    gausNoiseAcc_Gaus.append(superFunction(files[0], "gausNoise", nrows=100, nvalues=30, noise=gausNoiseDist[j], feature_cols=feature_cols, target=150, split=500))
    
for j in range(len(gausNoiseDist)):
    gausNoiseAcc_Uniform.append(superFunction(files[0], "gausNoise", nrows=100, nvalues=30, noise=gausNoiseDist[j], feature_cols=feature_cols, target=150, split=500))

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
    randSwapAcc_Gaus.append(superFunction(files[0], "randSwap", nrows=100, nvalues=randSwapDist[j], feature_cols=feature_cols, target=150, split=500))
    
for j in range(len(randSwapDist)):
    randSwapAcc_Uniform.append(superFunction(files[0], "randSwap", nrows=100, nvalues=randSwapDist[j], feature_cols=feature_cols, target=150, split=500))
    
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

