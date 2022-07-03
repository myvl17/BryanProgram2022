# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:25:27 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('MI.data', sep=',', header=None)

'''
FOR FUTURE REFERENCE:
HOW TO ADD ATTRIBUTE NAMES FROM ONLINE WEBSITE
ONLY USE IN DESPERATE SITUATIONS
----------------------------------------------

col_names = []

file = open('col_names.txt', 'r', encoding=("utf8"))
Lines = file.readlines()

import re

for i in range(len(Lines)):
    line = Lines[i]
    find = re.findall('\(.*?\)', line)
    
    
    if len(find) > 0:
        thisOne = find[0]
        thisOne = thisOne.replace('(', '').replace(')', '')
        col_names.append(thisOne)
    
print(col_names)
'''

col_names = ['ID', 'AGE', 'SEX', 'INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'IBS_POST', 'IBS_NASL', 'GB', 'SIM_GIPERT', 'DLIT_AG', 'ZSN_A', 'nr11', 'nr01', 'nr02', 'nr03', 'nr04', 'nr07', 'nr08', 'np01', 'np04', 'np05', 'np07', 'np08', 'np09', 'np10', 'endocr_01', 'endocr_02', 'endocr_03', 'zab_leg_01', 'zab_leg_02', 'zab_leg_03', 'zab_leg_04', 'zab_leg_06', 'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 'O_L_POST', 'K_SH_POST', 'MP_TP_POST', 'SVT_POST', 'GT_POST', 'FIB_G_POST', 'ant_im', 'lat_im', 'inf_im', 'post_im', 'IM_PG_P', 'ritm_ecg_p_01', 'ritm_ecg_p_02', 'ritm_ecg_p_04', 'ritm_ecg_p_06', 'ritm_ecg_p_07', 'ritm_ecg_p_08', 'n_r_ecg_p_01', 'n_r_ecg_p_02', 'n_r_ecg_p_03', 'n_r_ecg_p_04', 'n_r_ecg_p_05', 'n_r_ecg_p_06', 'n_r_ecg_p_08', 'n_r_ecg_p_09', 'n_r_ecg_p_10', 'n_p_ecg_p_01', 'n_p_ecg_p_03', 'n_p_ecg_p_04', 'n_p_ecg_p_05', 'n_p_ecg_p_06', 'n_p_ecg_p_07', 'n_p_ecg_p_08', 'n_p_ecg_p_09', 'n_p_ecg_p_10', 'n_p_ecg_p_11', 'n_p_ecg_p_12', 'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 'fibr_ter_05', 'fibr_ter_06', 'fibr_ter_07', 'fibr_ter_08', 'GIPO_K', 'K_BLOOD', 'GIPER_Na', 'Na_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'KFK_BLOOD', 'L_BLOOD', 'ROE', 'TIME_B_S', 'R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n', 'NA_KB', 'NOT_NA_KB', 'LID_KB', 'NITR_S', 'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n', 'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n', 'LID_S_n', 'B_BLOK_S_n', 'ANT_CA_S_n', 'GEPAR_S_n', 'ASP_S_n', 'TIKL_S_n', 'TRENT_S_n', 
             'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS']

data.columns = col_names




# chronic = data.drop(['FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'REC_IM', 'P_IM_STEN', 'LET_IS'], axis=1)

chronic = data.drop(['FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'REC_IM', 'P_IM_STEN', 'LET_IS',
                    'IBS_NASL', 'R_AB_3_n', 
                    'GIPO_K', 'K_BLOOD', 'NA_KB', 'NOT_NA_KB', 'LID_KB',
                    'S_AD_KBRIG', 'D_AD_KBRIG', 'KFK_BLOOD', ], axis=1)

pd.options.display.max_columns = chronic.shape[1]

# Finds all '?' and replaces with Na
for i in range(chronic.shape[0]):
    for j in range(chronic.shape[1]):
        if chronic.iloc[i, j] == '?':
            chronic.iloc[i, j] = None
            
# Replace all missing values
chronic = chronic.replace(to_replace='None', value=np.nan).dropna()
chronic = chronic.reset_index()
            
print(chronic.dtypes)

# chronic_corrMatrix = chronic.corr()
# sns.heatmap(chronic_corrMatrix, annot=False)
# plt.show()


