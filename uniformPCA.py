# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:28:01 2022

@author: cdiet
"""
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from generateRawData import generateRawData
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sn

hfont = {'fontname':'normal', 'size': 15}
vfont = {'fontname':'normal', 'size': 10}

font = {'family':'normal', 'weight':'bold', 'size':10}

plt.rc('font', **font)

cols = 150
rows = 500

distance = 0.5
df1 = generateRawData(rows, cols, distance, 'uniform')
label = df1[df1.shape[1] - 1]
feature_cols = df1.drop(columns = df1.shape[1] - 1)
df= StandardScaler().fit_transform(df1)

pca = PCA(n_components=150)
components = pca.fit_transform(feature_cols)

exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
exp_var = pca.explained_variance_ratio_

plt.plot(range(1, exp_var.shape[0] + 1), exp_var, linewidth = 7.0, label = "exp var", color = 'magenta')
plt.xlabel("Number Principle Components", **hfont)
plt.ylabel("Explained Variance", **hfont)
plt.plot(range(1, exp_var_cumul.shape[0] + 1), exp_var_cumul, linewidth = 7.0, label = "exp var cumul", color = 'blue')

plt.legend()
plt.show()


fig, ax = plt.subplots(1, 2, sharey = True, figsize = (10, 5))
ax[0].plot(range(1, exp_var.shape[0] + 1), exp_var, linewidth = 7.0, color = 'magenta', label = 'exp var')
ax[0].set_title("Explained Variance", **hfont)
ax[0].set_ylabel("Explained Variance", **hfont)
ax[0].set_xlabel("Principle Component", **hfont)
plt.legend()

ax[1].plot((range(1, exp_var_cumul.shape[0] + 1)), exp_var_cumul, linewidth = 7.0, color = 'blue', label = 'exp var cumul')
ax[1].set_title("Explained Variance Cumulative", **hfont)
ax[1].set_ylabel("Explained Variance Cumulative", **hfont)
ax[1].set_xlabel("Principle Component", **hfont)
plt.legend()
plt.tight_layout()
plt.show()

components2 = pd.DataFrame(components)

components3 = pd.concat([components2, label], axis = 1, ignore_index = True)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection = '3d')

ax1.scatter(components3.iloc[:, 0], components3.iloc[:, 1], components3.iloc[:, 2], c = components3[components3.shape[1] - 1])
ax1.set_xlabel('Principle Component 1', **vfont)
ax1.set_ylabel('Principle Component 2', **vfont)
ax1.set_zlabel('Principle Component 3', **vfont)
ax1.set_title('3 Principle Components', **vfont)

plt.show()

covMatrix = np.cov(components2.iloc[:25])
ax2 = plt.axes()
sn.heatmap(covMatrix, fmt='g')
ax2.set_title('Covariance Matrix 25 Principle Components')
plt.show()

# x = []
# for i in range(len(y)):
#     x.append(i)
    

# plt.plot(x, y)
# plt.show()

# import pandas as pd
# import plotly.express as px
# from sklearn.decomposition import PCA
# from sklearn.datasets import load_boston

# # boston = load_boston()
# # df = pd.DataFrame(boston.data, columns=boston.feature_names)
# n_components = 4
# df.rename(columns = {150:'target'}, inplace = True)

# pca = PCA(n_components=n_components)
# components = pca.fit_transform(df)

# total_var = pca.explained_variance_ratio_.sum() * 100

# labels = {str(i): f"PC {i+1}" for i in range(n_components)}
# labels['target'] = 'Median Price'

# fig = px.scatter_matrix(
#     components,
#     color= df['target'],
#     dimensions=range(n_components),
#     labels=labels,
#     title=f'Total Explained Variance: {total_var:.2f}%',
# )
# fig.update_traces(diagonal_visible=False)
# fig.show()


import plotly.express as px
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



# import plotly.express as px
# from sklearn.decomposition import PCA
# from sklearn import datasets
# from sklearn.preprocessing import StandardScaler

# df = px.data.iris()
# features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# X = df[features]

# pca = PCA(n_components=2)
# components = pca.fit_transform(X)

# loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# fig = px.scatter(components, x=0, y=1, color=df['species'])

# for i, feature in enumerate(features):
#     fig.add_shape(
#         type='line',
#         x0=0, y0=0,
#         x1=loadings[i, 0],
#         y1=loadings[i, 1]
#     )
#     fig.add_annotation(
#         x=loadings[i, 0],
#         y=loadings[i, 1],
#         ax=0, ay=0,
#         xanchor="center",
#         yanchor="bottom",
#         text=feature,
#     )
# fig.show()

