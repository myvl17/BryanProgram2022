# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:27:52 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from updatedSuperFunction import applyAugmentationMethod

def corrAugmentation(data, nrows, nvalues):
    
    # df[].corr(df[])

    for i in range(1):
        for j in range(data.shape[1]):
            print(str(i) + 'x' + str(j) + ' : ' + str(data.iloc[:, i].corr(data.loc[:, j])))

    col1 = data.iloc[:, 0]
    col2 = data.iloc[:, 1]
    col3 = data.iloc[:, 2]
    col4 = data.iloc[:, 3]
    col5 = data.iloc[:, 4]
    
    
    randRow = random.sample(range(0, col1.shape[0]-1), nvalues)
    
    for i in range(len(randRow)):
        if (random.randint(0, 1) == 1):
            col1.iloc[randRow[i]] += .1
            col2.iloc[randRow[i]] += .1
            col3.iloc[randRow[i]] += .1
            col4.iloc[randRow[i]] += .1
            col5.iloc[randRow[i]] += .1
            
        else:
            col1.iloc[randRow[i]] -= .1
            col2.iloc[randRow[i]] -= .1
            col3.iloc[randRow[i]] -= .1
            col4.iloc[randRow[i]] -= .1
            col5.iloc[randRow[i]] -= .1

    data[0] = col1
    
    
    
    
    for i in range(1):
        for j in range(data.shape[1]):
            print(str(i) + 'x' + str(j) + ' : ' + str(data.iloc[:, i].corr(data.loc[:, j])))
    
    return None


from generateRawData import generateRawData

data = generateRawData(10, 5, .25, 'uniform')

aug = corrAugmentation(data, 10, 5)


'''
0x0 : 1.0
0x1 : 0.32241484818158567
0x2 : -0.08488468694598805
0x3 : 0.2747705766105539
0x4 : 0.21223653504939402
0x5 : 0.4741174260130165

0x0 : 1.0
0x1 : 0.4213049480056305
0x2 : 0.011758040109942412
0x3 : 0.3951488287206735
0x4 : 0.2896545012953739
0x5 : 0.3998821004919267
'''