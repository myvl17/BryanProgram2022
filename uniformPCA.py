# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:28:01 2022

@author: cdiet
"""
# import numpy as np
# from sklearn.decomposition import PCA
# import pandas as pd
# import matplotlib.pyplot as plt
from generateRawData import generateRawData
# from sklearn.preprocessing import StandardScaler

cols = 150
rows = 500

distance = 0.5
df = generateRawData(rows, cols, distance, 'uniform')
feature_cols = df.drop(columns = df.shape[1] - 1)

# pca = PCA(n_components=3, svd_solver='full')
# pca.fit(data)
# y = pca.explained_variance_ratio_
# z = pca.singular_values_

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
df.rename(columns = {150:'target'}, inplace = True)

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


X = df.iloc[:,:4]

pca = PCA(n_components=3)
components = pca.fit_transform(X)

total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=df['target'],
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.show(method='external')

