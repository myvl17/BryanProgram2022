# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 09:25:41 2022

@author: cdiet
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fakeSuper import applyAugmentationMethod
from betterLogRegpy import betterLogReg
from columnNumberTesting import runClassifier
from evenBetterRandSwap import betterRandSwap
from betterGausNoise import betterGausNoise
from modifiedGausNoise import modifiedGausNoise



## Original filtering of dataset, explanation of a set of variables
# cols = ['CRKUS30A',  # Number of days used crack in past month, between 1 and 30, 91 = never
#     "COCUS30A", #  # Number of days used cocaine in past month, between 1 and 30, 91 = never
#     "ALCDAYS",  # Number of days  drank alcohol in past month, between 1 and 30, 91 = never
#     "CATAG3", # Age group: 1 = 12-17, 2 = 18-25, 3 = 26-34, 4 = 35-49, 5 = 50+
#     "HEALTH2", # Health condition: 1 = Excellent, 2 = very good, 3 = good, 4 = fair/poor 
#     "IRWRKSTAT", # Work status: 1 = full time, 2 = part time, 3 = unemployed, 99 = 12-14 yrs old
#     "IREDUHIGHST2", # Highest completed education
#     "IRSEX", # Sex: 1 = male, 2 = female
#     "IRPINC3", # Income range (thousands): 1 = <10, 2 = 10-20, 3 = 20-30, 4 = 30-40, 5 = 40-50, 6 = 50-75, 7 = 75+
#     "IRKI17_2", # Number of kids <18 y/o: 1 = none, 2 = 1, 3 = 2, 4 = 3+
#     "MRDAYPYR", # Number of days used marijuana in past year: 1-366, 991 = none
#     "WRKDHRSWK2", # Number of hours worked in past week: 1-60, 61 = 61+
#     "IRHHSIZ2", # Number of people in household: 1 - 6+
#     "CIG30USE", # Number of days smoked cigarettes in past month, between 1 and 30, 91 = never
#     "HEREVER", # Number of days used heroine in past year
#     "IRMARITSTAT", # Marital status: 1 = married, 2 = widowed, 3 = divorced, 4 = never married, 99 = <15
#     "DIABETEVR", # 1 = yes, 2 = no
#     "METHAMEVR" # 1 = yes, 2 = no
# ]


## Read in the file, it is separated by tabs
df = pd.read_csv('NSDUH_2015_Tab.tsv', sep = '\t', na_values = 0)

## Filter the dataset to only contain an answer yes or no to diabetes
dfdiab = df.loc[(df['DIABETEVR'] == 1) | (df['DIABETEVR'] == 2)]

## A look into the dataset through a histogram
# # plt.hist(x = df['IRMARITSTAT'], bins = [-9, -8, 1, 2, 3, 4, 5, 98, 99], rwidth = 0.5)

## Original filtering of few variables to make graphs interpretable
# # df = df.loc[(df['METHAMEVR'] == 1) | (df['METHAMEVR'] == 2)]
# # df = df[(df['CRKUS30A'] <= 30) | (df['CRKUS30A'] == 91)]
# # df = df[(df['COCUS30A'] <= 30) | (df['COCUS30A'] == 91) ]
# # df = df[(df['COCUS30A'] <= 30) | (df['COCUS30A'] == 91)]
# # df = df[(df['CIG30USE'] <= 30) | (df['CIG30USE'] == 91)]
# # df = df[(df['IRWRKSTAT'] <= 3) | (df['IRWRKSTAT'] == 99)]
# df = df[(df['MRDAYPYR'] <= 366) | (df['MRDAYPYR'] == 991)]
# df = df[(df['WRKDHRSWK2'] <= 61)]

## Important variables for running dataset experiments
ITERATIONS = 25

data = np.arange(25, 2000, 25)
rows = 150
cols = 100

## Filter dataset so 1 and for diabetes are 0 and 1, remove string column
dfdiab['DIABETEVR'] = dfdiab['DIABETEVR'].replace([1, 2], [0, 1])
dfdrop = dfdiab.drop(axis = 1, labels = "FILEDATE")

diab = dfdiab['DIABETEVR']

dfdrop2 = dfdrop.drop(axis = 1, labels = "DIABETEVR")


# # dfdrop2 = dfdrop2.rename(columns = cols)

## Filter more so no more strings and convert to float so more compatible
dffinal = pd.concat([dfdrop2, diab], axis = 1, ignore_index = True)

dffinal2 = dffinal.reset_index(drop = True)

diab2 = dffinal2[dffinal2.shape[1] - 1]

dffinal3 = dffinal2.replace({"C":None, 'O':None})

dfreal = dffinal3.astype(float)

dfZero = dfreal.replace([91, 991, 9991, 93, 993, 9993], 0)
dfOther = dfZero.replace([94, 9997, 97, 997, 98, 9998, 998, 9994, 994, 85, 9985, 985, 89, 989, 9989, 99, 9999, 999], "NaN")
# ITERATIONS2 = 100
# countAcc = 0
# accFin = []
# for a in range(1, ITERATIONS2):

# Imputation of missing values so no more nas
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan , strategy='mean')
imp.fit(dfOther)
dfImp = pd.DataFrame(imp.transform(dfOther))

dfInt = dfImp.astype(int)
dflast = dfInt.astype(float)
# np.savetxt('diabData.txt', dflast)

## Variables for running experiments
pmUnit = np.arange(0, 10 , 0.25)
randUnit = np.arange(0, cols, 1)
gausNoise = np.arange(0, 5.25, 0.25)
modGausNoise = np.arange(0, cols, 1)

rowIter = np.arange(0, 1050, 50)
counter = 0
count = 0
count2 = 0

# # df = pd.read_table('diabData.txt', header = None)

## Find the ratio of diabetes to no diabetes
for i in range(0, dflast.shape[0]):
    row = dflast.iloc[i, 2672] 
    if (row == 1):
        count += 1
    else:
        count2 += 1   
        
ratio = count/dflast.shape[0]   


## Create new dataframe with only 100 most important columns
## Use lowest p-value
dftest = pd.DataFrame()
from scipy.stats import ttest_ind

count5 = 0
for i in range(0, dflast.shape[1] - 2):
    ttest = ttest_ind(dflast[i], dflast[i + 1]) 
    if (ttest[1] == 0):     
        dftest = pd.concat([dftest, dflast.iloc[:, i]], axis = 1, ignore_index = True)
        count5 += 1
        if (count5 >= cols):
            break
   
## Add labels back to dataset         
dftest = pd.concat([dftest, diab2], axis = 1, ignore_index = True)



# # dftest = pd.concat([dflast.iloc[:rows, :cols], diab2[:rows]], axis = 1, ignore_index = True)

## Create new dataframe with right ratio along with 100 most important columns
count4 = 0
dftest2 = pd.DataFrame()
numbDiab = int(rows*ratio)

noDiab = dftest[dftest[cols] == 0]
yesDiab = dftest[dftest[cols] == 1]

df100 = pd.concat([yesDiab.iloc[:numbDiab, :], noDiab.iloc[:rows - numbDiab, :]], axis = 0, ignore_index= True)
df100 = (df100.sample(frac = 1)).reset_index(drop = True)

np.savetxt("diabData.txt", df100)
  
## BEGIN EXPERIMENTS



# # list1 = []
# # for i in range(dffinal2.shape[1]):
# #     # if (df.iloc[:, i].dtypes == "str"):
# #     #     dfdrop2 = dfdrop.drop(axis = 1, labels = dfdrop.columns[i])
# #     list1.append(dffinal2.iloc[:, i].dtypes)
    

    
    
    
# # list2 = []
# # for value in list1:
# #     if(value ==  )
        
# # dfreal = dfdrop2.astype(float)

# # plt.scatter(x = df["WRKDHRSWK2"], y = df['CIG30USE'])

# # cols = ['IRALCFY','WRKDHRSWK2']

# # # Function that cleans continuous numerical data counting by year
# # def cont_clean_data(x):

# #     # Survey codes for "Bad Data", Don't Know, Skip, Refused, or Blank
# #     if (x == -9) or (x == 985) or (x == 989) or ((x >= 994) and (x < 1000)):
# #         return np.nan

# #     # Codes for "Have never done..." or "Have not done in the past X days"
# #     # Equivalent to 0 for numbered questions
# #     if ((x == 991) or (x == 993)):
# #         return 0

# #     # Ignore value if conditions don't match
# #     return x 

# # df = df.applymap(cont_clean_data)


# # # Function that cleans all other special data codes
# # def ord_clean_data(x):

# #     # Survey codes for "Bad Data", Don't Know, Skip, Refused, or Blank
# #     if ((x == -9) |
# #     ((x >= 94) & (x < 100)) |
# #     (x == 85) |
# #     (x == 89)):
# #         return np.nan

# #     # Codes for "Have never done..." or "Have not done in the past X days"
# #     # Equivalent to 0 for numbered questions
# #     if ((x == 91) |
# #     (x == 93)):
# #         return 0

# #     # Ignore value if conditions don't match
# #     return x 

# # # Changes for wrkdhrsw2
# # df[(df['IRWRKSTAT'] == 3) | (df['IRWRKSTAT'] == 4), "WRKDHRSWK2"] = 0
# # df.loc[df['WRKDHRSWK2'] == 61, "WRKDHRSWK2"] = np.nan

# # # Changes for binary categorical variables
# # df.loc[(df['COCEVER'] == 2), "COCEVER"] = 0
# # df.loc[(df['CRKEVR'] == 2), "CRKEVR"] = 0
# # df.loc[(df['IRSEX'] == 2), "IRSEX"] = 0

# # Apply clean_data functions
# # df[cont_cols] = df[cont_cols].applymap(cont_clean_data)
# # df[ord_cols] = df[ord_cols].applymap(ord_clean_data)



# import seaborn as sns
# # # Function for easily plotting sns barplots on a grid
# # def plot_bar(data, grid, x, y, xlabel, ylabel, title, xticklabels, rotation=0):
# #     ax = fig.add_subplot(grid[0], grid[1], grid[2])
# #     sns.barplot(data=data, x=x, y=y, 
# #     estimator=(lambda x: sum(x)/len(x)), ax=ax).set_title(title)
# #     ax.set(xlabel=xlabel, ylabel=ylabel)
# #     ax.set_xticklabels(xticklabels, rotation=rotation)

# # # Set figure parameters
# # plt.rcParams['figure.figsize'] = [16, 12]
# # plt.rcParams['figure.subplot.wspace'] = 0.3
# # plt.rcParams['figure.subplot.hspace'] = 0.7
# # fig = plt.figure()

# # # Call plot_bar to plot bar graphs for various variables
# # plot_bar(df,[2, 2, 1], 'DIABETEVR', 'IRALCFY', 'Diabetes Diagnosis', 'Days Consumed Alcohol', 
# # "Average # Days Consumed Alcohol\nin a Year vs Diabetes Diagnosis", ["Has Diabetes", "Does Not Have Diabetes"])

# # # plot_bar(df,[2, 2, 2], 'DIABETEVR', 'WRKDHRSWK2', 'Diabetes Diagnosis', 'Hours Worked', 
# # # "Average # Hours Worked in a\nWeek vs Diabetes Diagnosis", ["Has Diabetes", "Does Not Have Diabetes"])

# # plot_bar(df,[2, 2, 2], 'DIABETEVR', 'IRPINC3', 'Diabetes Diagnosis', 'Income Range', 
# # "Average Income Range\n vs Diabetes Diagnosis", ["Has Diabetes", "Does Not Have Diabetes"])

# # plot_bar(df,[2, 2, 3], 'DIABETEVR', 'IRSEX', 'Diabetes Diagnosis', 'Sex', 
# # "Sex vs Diabetes Diagnosis", ["Has Diabetes", "Does Not Have Diabetes"])

# # plot_bar(df,[2, 2, 4], 'DIABETEVR', 'CATAG3', 'Diabetes Diagnosis', 'Age Group', 
# # "Average Age Group\n vs Diabetes Diagnosis", ["Has Diabetes", "Does Not Have Diabetes"])

# # Stolen from Jeff
# # Exploration graphs
# # fig, ax = plt.subplots(2, sharey=False)
# # g = sns.histplot(df, x='IRSEX', hue='DIABETEVR', multiple='dodge', ax=ax[0]).set(title='Sex', xlabel = 'Sex')

# # sns.histplot(df, hue='DIABETEVR', x='CATAG3', multiple='dodge', ax=ax[1]).set(title='Age Category', xlabel = 'Age')

# # plt.tight_layout()
# # plt.show()

# # plot_bar(df,[2, 2, 2], 'ireduhighst2', 'coccrkever', 'Highest Completed Education', 'Proportion of People Who\nHave Used Crk/Cocaine', 
# # "Proportion of People who have\nUsed Crk/Coc by Highest Completed Education",
# # ["5th or less", "6th", "7th", "8th", "9th", "10th", "11th/12th,\nno diploma,", 
# # "High school\ndiploma/GED", "Some college,\nno degree", "Associate's Deg.", "College Grad\nor Higher"], 90)
# # print(df.corr())

# # X = df.drop(df["DIABETEVR"])   #Feature Matrix
# # y = df["DIABETEVR"]          #Target Variable
# # df.head()

# # plt.figure(figsize=(12,10))
# # cor = df.corr()
# # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# # plt.show()

