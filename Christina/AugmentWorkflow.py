# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:16:47 2022

@author: cdiet
"""

import pandas as pd
import numpy as np

# from SyntheticData import UniformSythetic

# UniformSythetic(500, 150, 2)

# from UniformAugmentation import RandUnit

# # Run the function
# print(RandUnit('synthetic_data_with_labels.txt', 500, 0.1, 150))   
       

# from LogisticRegressionReal import LogReg

# feature_cols = []
# for i in range(0, 149, 1):
#     feature_cols.append(i)
# print(LogReg(dataset = 'augmented_original_label.txt',
#               feature_cols = feature_cols, target = 'status', split = 500,
#               save = 'augmented_data_labels.txt'))

# Take the labels from the original data, append the predicted labels
# Add that column to original and augmented data

data = pd.read_table('augmented_original.txt', delimiter = " ", header = None)
original_label = pd.read_table('synthetic_data_labels.txt', delimiter = " ", header = None)

augmented_label = pd.read_table('augmented_data_labels.txt', delimiter = " ", header = None)

labels =  pd.concat([original_label, augmented_label])

data['status'] = labels

# Output to txt
np.savetxt('all_data.txt', data)