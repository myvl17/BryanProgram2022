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

"""
Jason's K-Nearest Neighbor
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
 
knn = KNeighborsClassifier(n_neighbors=7)
 
knn.fit(X_train, y_train)

# Predict on dataset which model has not seen before
print(knn.predict(X_test))
 
# Calculate the accuracy of the model
print(knn.score(X_test, y_test))
