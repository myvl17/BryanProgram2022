# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:49:05 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generateRawData import generateRawData
from columnNumberTesting import runClassifier


numRows = np.arange(50, 550, 50)
numCols = np.arange(50, 550, 50)
acc = []

for i in numRows:
    fig = plt.subplots(figsize=(30,20))
    for j in numCols:
        
        distance = np.arange(0, 2, .25)
        dAcc = []
        
        for k in range(len(distance)):
            data = generateRawData(i, j, distance[k], 'gaussian')
            dAcc.append(runClassifier(data, 'SVM', 'f1'))
            

        plt.plot(distance, dAcc, label=j, linewidth=5)
            
        plt.title(str(i) + ' rows', fontdict=({'fontsize':50}))
        plt.xticks(distance, fontsize=50)
        plt.xlabel('Distance', fontsize=50)
        plt.yticks(np.arange(.3, 1.1, .1), fontsize=50)
        plt.ylabel('Accuracy', fontsize=50)
        plt.grid(True, alpha=0.8)
        plt.legend(fontsize=30, title='Column Amount', title_fontsize=30,fancybox=True)
        plt.tight_layout()
        # plt.style.use('dark_background')
    plt.show()
        
        





