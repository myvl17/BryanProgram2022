# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:20:34 2022

@author: jeffr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Maintains random data set generated
np.random.seed(1)

# Class 1 normal distribution
df0 = pd.DataFrame(np.random.normal(5, 1, size=(250, 150)))
df0["Status"] = 0

# Class 2 normal distribution
df1 = pd.DataFrame(np.random.normal(15, 1, size=(250, 150)))
df1["Status"] = 1

# Merging both data sets
dataset = pd.concat([df0, df1])

# Shuffling data set
df = pd.DataFrame(np.random.permutation(dataset))


plt.scatter(df[0], df[1])
plt.show()

np.savetxt('Gaussian Distribution Data Set with Status.txt', df)

df.rename(columns = {150: 'status'}, inplace = True)


feature_cols = []
for i in range(150):
    feature_cols.append(i)
    
X = df[feature_cols]

y = df['status']


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
