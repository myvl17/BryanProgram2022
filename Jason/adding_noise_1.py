# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(1)

df0 = pd.DataFrame(np.random.normal(5, 1, size=(250, 150)))
df1 = pd.DataFrame(np.random.normal(15, 1, size=(250, 150)))


dataset = pd.concat([df0, df1])
df = pd.DataFrame(np.random.permutation(dataset))
print(df)



plt.scatter(df[0], df[1])
plt.show()

np.savetxt('Gaussian Distribution Data Set.txt', df)

data_copy = pd.DataFrame.copy(df)


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
    
aug_data = (adding_noise(data_copy, 7, 0.05))
print(aug_data)