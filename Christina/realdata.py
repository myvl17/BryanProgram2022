# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 09:25:41 2022

@author: cdiet
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cols = ['CRKUS30A',  # Number of days used crack in past month, between 1 and 30, 91 = never
    "COCUS30A", #  # Number of days used cocaine in past month, between 1 and 30, 91 = never
    "ALCDAYS",  # Number of days  drank alcohol in past month, between 1 and 30, 91 = never
    "CATAG3", # Age group: 1 = 12-17, 2 = 18-25, 3 = 26-34, 4 = 35-49, 5 = 50+
    "HEALTH2", # Health condition: 1 = Excellent, 2 = very good, 3 = good, 4 = fair/poor 
    "IRWRKSTAT", # Work status: 1 = full time, 2 = part time, 3 = unemployed, 99 = 12-14 yrs old
    "IREDUHIGHST2", # Highest completed education
    "IRSEX", # Sex: 1 = male, 2 = female
    "IRPINC3", # Income range (thousands): 1 = <10, 2 = 10-20, 3 = 20-30, 4 = 30-40, 5 = 40-50, 6 = 50-75, 7 = 75+
    "IRKI17_2", # Number of kids <18 y/o: 1 = none, 2 = 1, 3 = 2, 4 = 3+
    "MRDAYPYR", # Number of days used marijuana in past year: 1-366, 991 = none
    "WRKDHRSWK2", # Number of hours worked in past week: 1-60, 61 = 61+
    "IRHHSIZ2", # Number of people in household: 1 - 6+
    "CIG30USE", # Number of days smoked cigarettes in past month, between 1 and 30, 91 = never
    "HEREVER", # Number of days used heroine in past year
    "IRMARITSTAT", # Marital status: 1 = married, 2 = widowed, 3 = divorced, 4 = never married, 99 = <15
    "DIABETEVR", # 1 = yes, 2 = no
    "METHAMEVR" # 1 = yes, 2 = no
]

df = pd.read_csv('NSDUH_2015_Tab.tsv', sep = '\t', usecols = cols)

df = df.loc[(df['DIABETEVR'] == 1) | (df['DIABETEVR'] == 2)]

descrip = df.describe()

# plt.hist(x = df['IRMARITSTAT'], bins = [-9, -8, 1, 2, 3, 4, 5, 98, 99], rwidth = 0.5)

df = df.loc[(df['METHAMEVR'] == 1) | (df['METHAMEVR'] == 2)]
df = df[(df['CRKUS30A'] <= 30) | (df['CRKUS30A'] == 91)]
df = df[(df['COCUS30A'] <= 30) | (df['COCUS30A'] == 91) ]
df = df[(df['COCUS30A'] <= 30) | (df['COCUS30A'] == 91)]
df = df[(df['CIG30USE'] <= 30) | (df['CIG30USE'] == 91)]
df = df[(df['IRWRKSTAT'] <= 3) | (df['IRWRKSTAT'] == 99)]
df = df[(df['MRDAYPYR'] <= 366) | (df['MRDAYPYR'] == 991)]
df = df[(df['WRKDHRSWK2'] <= 61)]
df = df.replace(to_replace =  (df['DIABETEVR'] == 2), value = 1)

plt.scatter(x = df["WRKDHRSWK2"], y = df['CIG30USE'])



print(df.corr())

# X = df.drop(df["DIABETEVR"])   #Feature Matrix
# y = df["DIABETEVR"]          #Target Variable
# df.head()

# plt.figure(figsize=(12,10))
# cor = df.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()