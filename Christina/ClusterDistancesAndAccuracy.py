# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:37:01 2022

@author: cdiet
"""

f1_accuracy = skm.f1_score(y_test, predicted_values)
results_df = results_df.append({'F1 Score' : f1_accuracy}, ignore_index=True)

from Mini_super_function import OkayFunction