# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:25:27 2022

@author: jeffr
"""

import numpy as np
import pandas as pd

data = pd.read_csv('MI.data', sep=',', header=None)

col_names = []

file = open('col_names.txt', 'r', encoding=("utf8"))
Lines = file.readlines()

for i in range(len(Lines)):
    line = Lines[i]
    test = str(i) + '.'
    
    if line[:2] == test:
        print(line[2:])
        print("YES")