# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:43:28 2022

@author: cdiet
"""
import pandas as pd
import matplotlib.pyplot as plt

# Create graph

original = pd.read_table('synthetic_data.txt', delimiter = " ", header = None)
augmented = pd.read_table('augmented_data.txt', delimiter = " ", header = None)

ax, fig = plt.subplots()
 
plt.scatter(original[0], original[1], label = "original_data") 
plt.scatter(augmented[0], augmented[1], alpha = 0.2, c = "red", label = "augmented_data") 
plt.title("Random Uniform Data with Augmented Data")
plt.legend()
plt.show()