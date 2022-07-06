# -*- coding: utf-8 -*-
"""
Created on Tue Jul 5 15:00:37 2022

@author: cdiet
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

x = [.25, .75, .4, .8, .9, 1, 1.1, .95, .925, .85]
y = [.3, .2, .75, .6, .75, .5, .75, .9, 1, 1.1]
 
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

df = pd.DataFrame({0:x, 1:y, 2:labels})

def correlationAugmentation(df):

    i = 0
    for i in range(df.shape[1] - 1):
        reg = LinearRegression().fit(df[i], df[i + 1])
        reg.score(X, y)
    
        return reg
    
    

print(correlationAugmentation(df))