# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 07:10:31 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


'''
1. age - age in years

2. sex - sex (1 = male; 0 = female)

3. cp - chest pain type (0 = asymptomatic, 1 = typical angina; 2 = atypical angina; 3 = non-anginal pain)

4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)

5. chol - serum cholestoral in mg/dl

6. fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)

7. restecg - resting electrocardiographic results (1 = normal; 2 = having ST-T wave abnormality; 0 = hypertrophy)

8. thalach - maximum heart rate achieved

9. exang - exercise induced angina (1 = yes; 0 = no)

10. oldpeak - ST depression induced by exercise relative to rest

11. slope - the slope of the peak exercise ST segment (2 = upsloping; 1 = flat; 0 = downsloping)

12. ca - number of major vessels (0-3) colored by flourosopy

13. thal - 2 = normal; 1 = fixed defect; 3 = reversable defect

14. num - the predicted attribute - diagnosis of heart disease (angiographic disease status) (Value 0 = < diameter narrowing; Value 1 = > 50% diameter narrowing)
'''

'''
NEED TO FIX:
thal, values of 0 are considered NULL
'''

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

for i in range(df.shape[0]):
    if df.loc[i, 'vessels'] == 4:
        df.loc[i, 'vessels'] = None


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


# Exploration graphs
fig, ax = plt.subplots(2,2, sharex=False, sharey=False)
g = sns.histplot(df, x='sex', hue='target', multiple='stack', stat='probability', ax=ax[0,0]).set(title='gender')

sns.histplot(df, hue='target', x='chest_pain', multiple='dodge', ax=ax[0,1]).set(title='chest pain')

sns.histplot(df, hue='target', x='angina', multiple='dodge', ax=ax[1,0]).set(title='angina')

sns.histplot(df, hue='target', x='vessels', multiple='dodge', ax=ax[1,1]).set(title='vessels')

plt.tight_layout()
plt.show()


positive = df[df['target'] == 1]
negative = df[df['target'] == 0]

positive_filtered = positive.filter(['age', 'sex', 'blood_pressure', 'cholestoral', 'heart_rate', 'st_depression', 'target'])
negative_filtered = negative.filter(['age', 'sex', 'blood_pressure', 'cholestoral', 'heart_rate', 'st_depression', 'target'])

print("OVERVIEW")
print(df[df['target'] == 1].describe().T)
print(df[df['target'] == 0].describe().T)

print("\n MALE")
print(positive_filtered[positive_filtered['sex'] == 1].describe().T)
print(negative_filtered[negative_filtered['sex'] == 1].describe().T)

print("\n FEMALE")
print(positive_filtered[positive_filtered['sex'] == 0].describe().T)
print(negative_filtered[negative_filtered['sex'] == 0].describe().T)

sns.boxplot(data=df, x='target', y='cholestoral', hue='sex').set(title='Cholestoral')
plt.show()
sns.boxplot(data=df, x='target', y='blood_pressure', hue='sex').set(title='Blood Pressure')
plt.show()


fig, ax = plt.subplots(2,2, sharex=False, sharey=False)
sns.histplot(data=df, x='age', hue='target', multiple='stack', ax=ax[0,0])

sns.histplot(data=df, x='blood_pressure', hue='target', multiple='stack', ax=ax[0,1])

sns.histplot(df, x='cholestoral', hue='target', multiple='stack', ax=ax[1,0])

sns.histplot(df, x='heart_rate', hue='target', multiple='stack', ax=ax[1,1])

plt.tight_layout()
plt.show()

