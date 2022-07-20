#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:36:51 2022

@author: jasonwhite
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import random
import sklearn.metrics as skm
from sklearn.datasets import make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sea

np.random.seed(1)

df = pd.read_csv("epilepsy_data.csv")



df_1 = df.loc[1: 400]
#print(df_1)




for i in range(0, len(df_1)):
    if df_1.iloc[i, df_1.shape[1]-1] > 1:
        df_1.iloc[i, df_1.shape[1]-1] = 0


df_1.to_csv('Binary_epilepsy.csv')

df_2 = df_1.sample(n = 10, ignore_index = True)


df_3 = df_2.drop(columns = ['Unnamed: 0'])

#print(df_3)

x = []
for i in range(178):
    x.append(i)    


fig, axs = plt.subplots(3)

axs[0].plot(x, df_3.iloc[9, :df_3.shape[1]-1], c = 'blue', label = 'eyes-closed activity')

axs[1].plot(x, df_3.iloc[8, :df_3.shape[1]-1], c = 'red', label = 'seizure activity')
axs[1].set_ylabel("Microvolt (µV)")
axs[2].plot(x, df_3.iloc[7, :df_3.shape[1]-1], label = 'eyes-open activity')

axs[0].legend(loc="upper right")
axs[1].legend(loc  ="upper right")
axs[2].legend(loc ="upper right")
plt.xlabel("Portion of Time (1 second)")
axs[1].set_xticks([])
axs[0].set_xticks([])
axs[2].set_xticks([])

axs[0].set_ylim(-500, 500)
axs[2].set_ylim(-500, 500)
axs[1].set_ylim(-500, 800)

axs[0].set_yticks([-500, -100,  100, 500], minor = False)
axs[2].set_yticks([-500, -100,  100, 500], minor = False)


plt.show()

