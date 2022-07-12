# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:27:56 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from generateRawData import generateRawData

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

plt.rc('font', **font)

hfont = {'fontname':'normal', 'size':15}



ROW_SIZE = 500

COL_SIZE = 150


data = generateRawData(ROW_SIZE, COL_SIZE, 1, 'gaussian')

plt.scatter(data[0], data[1], c=data[data.shape[1]-1])
plt.show()

X = data.iloc[:, :data.shape[1]-1]
Y = data.iloc[:, data.shape[1]-1]

X = StandardScaler().fit_transform(X)



pcaExp = PCA(n_components=COL_SIZE, svd_solver='full')
test = pcaExp.fit_transform(X)

arr = pcaExp.explained_variance_ratio_

arrSum = np.asarray(pcaExp.explained_variance_ratio_).cumsum()

total = pcaExp.explained_variance_ratio_.sum() * 100
print(total)


npArr = np.asanyarray(arr)

x = []
for j in range(len(arr)):
    x.append(j)
    
plt.plot(x, npArr, c='r', linewidth=10, label='Single Exp. Var.')
plt.plot(x, arrSum, c='black', linewidth=10, label='Cummulative Exp. Var.')
plt.ylabel("Explained Variance", **hfont)
plt.xlabel('# of Principle Components', **hfont)
plt.legend()
plt.show()


fig, ax = plt.subplots(2,1, figsize=(10, 10), sharex=True)

ax[0].plot(x, arrSum, c='black', linewidth=10)
ax[0].set_title("Cumulative Exp. Var.", **hfont)
ax[0].set_ylabel("Explained Variance", **hfont)
ax[1].plot(x, npArr, c='r', linewidth=10)
ax[1].set_title("Single Exp. Var.", **hfont)
ax[1].set_xlabel("# of Principle Components", **hfont)
ax[1].set_ylabel('Explained Variance', **hfont)
plt.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1,2, figsize=(20, 5), sharey=True)

ax[0].plot(x, arrSum, c='black', linewidth=10)
ax[0].set_title("Cumulative Exp. Var.", **hfont)
ax[0].set_ylabel("Explained Variance", **hfont)
ax[0].set_xlabel('# of Principle Components', **hfont)
ax[1].plot(x, npArr, c='r', linewidth=10)
ax[1].set_title("Single Exp. Var.", **hfont)
ax[1].set_xlabel("# of Principle Components", **hfont)
plt.tight_layout()
plt.show()



import seaborn as sns
# Creates correlation matrix

covMatrix = np.cov(test[:25, :])

ax = plt.axes()
sns.heatmap(covMatrix, annot=False, ax=ax)
ax.set_title("Covalence Matrix of First 25 Principle Components")
  
# displaying heatmap
plt.show()


























data = generateRawData(ROW_SIZE, COL_SIZE, 1, 'gaussian')


X = data.iloc[:, :data.shape[1]-1]
Y = data.iloc[:, data.shape[1]-1]

X = StandardScaler().fit_transform(X)

pcaExp = PCA(n_components=3, svd_solver='full')
test = pcaExp.fit_transform(X)

arr = pcaExp.explained_variance_ratio_

arrSum = np.asarray(pcaExp.explained_variance_ratio_).cumsum()

total = pcaExp.explained_variance_ratio_.sum() * 100
print(total)


npArr = np.asanyarray(arr)



# Create a new dataset from principal components 
df = pd.DataFrame(data = test, columns = ['PC1', 'PC2', 'PC3'])

target = pd.Series(data[data.shape[1]-1], name='target')

result_df = pd.concat([df, target], axis=1)

# Visualize Principal Components with a scatter plot
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection="3d")
# ax.set_title(total)
ax.set_xlabel('First Principal Component ', fontsize = 15)
ax.set_ylabel('Second Principal Component ', fontsize = 15)
ax.set_zlabel('Third Principal Component', fontsize=15)
# ax.set_title('Principal Component Analysis (2PCs) for Iris Dataset', fontsize = 20)

targets = [0,1]
colors = ['r', 'g']
for target, color in zip(targets, colors):
    indicesToKeep = data[data.shape[1]-1] == target
    ax.scatter(result_df.loc[indicesToKeep, 'PC1'], 
               result_df.loc[indicesToKeep, 'PC2'], 
               result_df.loc[indicesToKeep, 'PC3'],
               c = color, 
               s = 50)
ax.legend(['class 0', 'class 1'])
ax.grid()
plt.show()
