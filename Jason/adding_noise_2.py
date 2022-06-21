# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_table("Gaussian_Distribution.txt", delimiter=" ", header=None)

def adding_noise (data, n_rows = int, g_noise = int):
    #Create a noise matrix
    noise_matrix = pd.DataFrame(np.random.normal(0, g_noise, size = (n_rows, 150)))
    #Add noise to dataset if equal length
    if len(data) == n_rows:
        return (data.add(noise_matrix, fill_value = 0))
   
    #add noise to random rows matrix from data set
    else:
        data_portion = data.sample(n = n_rows, ignore_index=True)
        print(data_portion)
        print(noise_matrix)
        
        return data_portion.add(noise_matrix, fill_value = 0)
    
aug_data = (adding_noise(df, 7, 0.05))
final_data = aug_data.iloc[: , :-1]
print(final_data)
combined_data = pd.concat([df, final_data])
print(combined_data)
np.savetxt('Gaussian_noise.txt', final_data)
np.savetxt('combined_data.txt', combined_data)