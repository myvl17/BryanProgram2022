# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:19:38 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

<<<<<<< Updated upstream
df = pd.read_csv('datasets/heart.csv')

# Rename columns to understandable names
cols = ['age', 
        'sex', 
        'chest_pain', 
        'blood_pressure', 
        'cholestoral', 
        'fasting_blood_sugar', 
        'ecg', 
        'heart_rate', 
        'angina', 
        'st_depression', 
        'peak_st', 
        'vessels', 
        'thalassemia', 
        'target']
df.columns = cols


# Creates correlation matrix
corrMatrix = df.corr()

# Applies absolute value to entire matrix
for i in range(corrMatrix.shape[0]):
    for j in range(corrMatrix.shape[1]):
        corrMatrix.iloc[i, j] = abs(corrMatrix.iloc[i,j])

# Mask created to form diagonal heatmap
mask = np.triu(np.ones_like(corrMatrix, dtype=bool))

# Creation of heatmap aka correlation matrix
f, ax = plt.subplots(figsize=(11,9))

cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corrMatrix, mask=mask, cmap=cmap, vmax=.9, center=0, square=True, linewidth=.5, cbar_kws={'shrink':.5}, annot=True)
plt.show()


for i in range(2):
    induced_angina = df[df['angina'] == i]
    print('Percent of positive target where induced_angina = ', i, ': ', (induced_angina.shape[0] - induced_angina[induced_angina['target'] == 1].shape[0]) / induced_angina.shape[0])


positive = df[df['target'] == 1]
negative = df[df['target'] == 0]

plt.hist(positive['age'], density=False, bins=[10, 20, 30, 40, 50, 60, 70, 80, 90])
plt.show()

ax = positive['sex'].value_counts().plot(kind='bar', xlabel='sex', ylabel='number of positive targets')
ax.set_xticklabels(['M', 'F'], rotation=0)
plt.show()


positive = positive.filter(['blood_pressure', 'cholestoral', 'ecg', 'heart_rate', 'target'])
negative = negative.filter(['blood_pressure', 'cholestoral', 'ecg', 'heart_rate', 'target'])
print(positive.describe())
print(negative.describe())

plt.box(positive, x='cholesterol', y='target')
=======

df = pd.read_csv('datasets/heart.csv')

print(df.describe())

corrMatrix = df.corr()

sns.heatmap(corrMatrix, vmin=corrMatrix.values.min(), vmax=1, square=True, linewidths=0.1, annot=True, annot_kws={"fontsize":5})  

plt.tight_layout()
plt.show()


sns.regplot(x=df['thalachh'], y=df['output'], data=df, logistic=True, ci=None)
>>>>>>> Stashed changes
