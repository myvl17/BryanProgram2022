# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:08:05 2022

@author: jeffr
"""

import pandas as pd
import numpy as np
import random


def addGaussianNoise(file, perturbations, noise):
    df = pd.read_table(file, delimiter=" ", header=None)
    
    # Creates empty data frame
    augmented_df = pd.DataFrame()
    
    for i in range(perturbations):
        rand_noise = np.random.normal(0, noise, size=(df.shape[1]))
        random_row = random.randint(0, df.shape[0]-1)
        
        print(random.sample)
    
    
    
augmented = addGaussianNoise("Generated Gaussian Distribution.txt", 7, .05)