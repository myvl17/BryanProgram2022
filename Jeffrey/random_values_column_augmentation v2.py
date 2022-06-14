# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:41:43 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


df = pd.read_table("Gaussian Distribution Data Set with Status.txt", delimiter=" ", header=None)

df.rename(columns = {150: 'status'}, inplace = True)

def rand_value_col(perturbation):
    
    # Copies original data set
    augmented_df = df.copy(deep=True)
    
    for k in range(0, perturbation, 1):
               
        # Selects random row index
        random_row = random.randint(0, augmented_df.shape[0]-1)
        row_status = augmented_df.iloc[random_row, -1]
        
        # Adds new row from pre-existing random row
        augmented_df = pd.concat([augmented_df, augmented_df.iloc[[random_row]]], ignore_index=True)
        
        # Filter data set for common status
        temp = augmented_df[augmented_df['status'] == row_status]   
        
        
        # Performs 30 pertubations
        for i in range(30):
            
            # Selects random column index
            random_col = random.randint(0, augmented_df.shape[1]-2)
            
            # Selects random value from row and column while maintaining status
            rand_value = temp.iloc[random.randint(0, temp.shape[0]-1)][random_col]
            
            # Selects random index location and changes value
            augmented_df.iloc[-1][random_col] = rand_value # THIS BREAKS EVERYTHING
            
    return augmented_df

 
test = rand_value_col(1)


fig, ax = plt.subplots(1,2, sharey=True) #figsize=(50,20)

ax[0].scatter(df[1], df[2], alpha=0.3)

ax[1].scatter(test[1], test[2], alpha=0.3)

# ax[0].hist(df, density=True, bins=50)
# ax[1].hist(test, density=True, bins=50)
plt.show()

# print(df.describe(include='all'))
# print(test.describe(include='all'))


feature_cols = []
for i in range(150):
    feature_cols.append(i)
    
X = test[feature_cols]

y = test['status']



from sklearn.model_selection import train_test_split

# Splitting the sets
# Test_size indicates the ratio of testing to training data ie 0.25:0.75
# Random_state indicates to select data randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(max_iter = 10000)

# fit the model with data
logreg.fit(X_train,y_train)

# create the prediction
y_pred= logreg.predict(X_test)


# Confusion matrix
# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

# Visualize the confusion matrix using a heat map
class_names = [0, 1] # name  of classes

# initialize the plot
fig, ax = plt.subplots()

# Plot the tick marks
tick_marks = np.arange(len(class_names))

# Both take ticks and labels
plt.xticks(ticks = tick_marks, labels = class_names)
plt.yticks(ticks = tick_marks, labels = class_names)

# create heatmap with labels, specify colors in cmap
# Annot writes the numbers in the cells
# fmt set to g is for integers, d is decimal, s is string
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="BuPu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()

# Add labels for readability
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# On the heat map, the top left is a true positive(8), top right
# is a false positive(3), bottom left is a false negative(1), and bottom
# right is a true negative(37)

# Print out an evaluation of the model using accuracy,
# precision, and recall

# From all positive and negative cases (true and false), this
# is the amount we predicted correctly
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# From all the classes we predicted as positive (true positive and false
# positive), the amount that are actually positive
print("Precision:", metrics.precision_score(y_test, y_pred))



# From all of the positive classes (true positive and false negative),
# this is the amount we predicted correctly
print("Recall:",metrics.recall_score(y_test, y_pred))


# Create new plot for second figure
fig, ax2 = plt.subplots()

# Finds the probability estimates, takes a vector
y_pred_proba = logreg.predict_proba(X_test)[::,1]

# Creates a graph of Receiver Operating Characteristic, graph of true
# positive against false positive
# Takes in the true values, and predicted values
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc = "+str(auc))
plt.legend(loc=4)
plt.show()  
