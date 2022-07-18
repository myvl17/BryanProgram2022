# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:53:00 2022

@author: cdiet
"""

from sklearn.model_selection import train_test_split
def betterLogReg(dataset, feature_cols, target, split):
        
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
    logreg = LogisticRegression(max_iter = 10000)
    
    # fit the model with data
    
    #print(y_train)
    logreg.fit(X_train,y_train)
    
    # create the prediction
    y_pred= logreg.predict(X_test)

    # Appends predicted labels to NAN
    for i in range(split, dataset.shape[0]):
        dataset.loc[i, target] = y_pred[i - split]
        
    
    # plt.scatter(dataset[0], dataset[1], c = dataset[dataset.shape[1] - 1])
    return dataset
