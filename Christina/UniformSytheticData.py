# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:24:23 2022

@author: cdiet
"""
# Create uniformly distributed data and augment

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import random



# Create the function that utilizes the sythetic data
def RandUnit(dataset, numbRows, unit):
    
    df = pd.read_table(dataset, delimiter = " ", header = None) 
    

# if statement to determine if the number of rows entered is odd
# The sample function takes random rows from the df
# in this case it take in the NumbRows and the # of rows
    if (numbRows % 2 == 0):
        sample1 = df.sample(n = int(numbRows / 2))
        sample2 = df.sample(n = int(numbRows / 2))
    else:
        sample1 = df.sample(n = int((numbRows / 2 ) + 0.5))
        sample2 = df.sample(n = int((numbRows / 2) - 0.5))
        
# Reset the index in each sample so they increase from 0 to NumbRows        
    sample1real = sample1.reset_index(drop = True)
    sample2real = sample2.reset_index(drop = True)
    
# Create a list of random numbers
    randomlist = []
    for j in range(0, numbRows):
        n = random.randint(0, 149)
        randomlist.append(n)
        
# Select one of the random rows then use the random list to 
# pinpoint one specfic number in the dataframe and add or 
# subtract the unit specified in the function
    for i in range(len(sample1real)):
        for j in randomlist:
            oldValue = (sample1real.iloc[i, j])
            newValue = oldValue + unit
            # Replace the oldvalue with the new value in the
            # samples set
            sample1real.replace(to_replace = oldValue, value = newValue)
       
    for i in range(len(sample2real)):
        for j in randomlist:
            oldValue = (sample2real.iloc[i, j])
            newValue = oldValue - unit
            sample2real.replace(to_replace = oldValue, value = newValue)
    
    # Add all new rows to the existing dataframe
    dffinaltest = pd.concat([df, sample1real, sample2real])

    # Reset the index again so it increases from 0 to n
    dffinal = dffinaltest.reset_index(drop = True)
    
# Create a list of values where it loops through each row 
# and determines if the first value is between 0 and 1 or 3
# and 4 and then add the list as a column to the dataframe
    targetvalue = []
    for i in range(len(dffinal)):
        if ((dffinal.iloc[i, 0]) < 1 and (dffinal.iloc[i, 0]) > 0):
            targetvalue.append(0)
        else:
            targetvalue.append(1)
            
    dffinal['status'] = targetvalue
    
    # Add too the original scatterplot with a different
    #alpha to show the new points
    plt.scatter(dffinal[0], dffinal[1], alpha = 0.5) 
    plt.show()

    # Save dataframe as a text file to be used outside
    # of this function
    np.savetxt('dataframe.txt', dffinal)

    return dffinal
    
# Run the function
print(RandUnit('test.txt', 500, 0.1))   
       

# LOGISTIC REGRESSION CODE

# load in the data and rename the status column
dffinal2 = pd.read_table('dataframe.txt', delimiter = " ", header = None)
dffinal2.rename(columns = {150: 'status'}, inplace = True)

# import pandas
import seaborn as sns

# Loop through to create a list of the names for the feature cols
feature_cols = []
for i in range(0, 149, 1):
    feature_cols.append(i)
    

# Feature variables
X = dffinal2[feature_cols]

# Target variable
y = dffinal2['status']

# Split both x and y into training and testing sets

# Import train_test_split
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
# 91.8% accuracy, really good

# From all the classes we predicted as positive (true positive and false
# positive), the amount that are actually positive
print("Precision:", metrics.precision_score(y_test, y_pred))
# Model predicted a person has parkinson's and they actually do
# 92.5% of the time

# From all of the positive classes (true positive and false negative),
# this is the amount we predicted correctly
print("Recall:",metrics.recall_score(y_test, y_pred))
# People who have Parkinson's can be identified 97.4% of the time

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

# auc of 1 is a perfect classifier and auc of 0.5 is a 
#worthless classifier, auc = 0.933
    


