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

# Imputation of missing values so no more nas
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(dfreal)
dflast = pd.DataFrame(imp.transform(dfreal))
# np.savetxt('diabData.txt', dflast)

## Variables for running experiments
pmUnit = np.arange(0, 10 , 0.25)
randUnit = np.arange(0, cols, 5)
gausNoise = np.arange(0, 5, 0.1)
modGausNoise = np.arange(0, cols, 5)

rowIter = np.arange(50, 1050, 50)
counter = 0
count = 0
count2 = 0

# # df = pd.read_table('diabData.txt', header = None)

## Find the ratio of diabetes to no diabetes
for i in range(0, dflast.shape[0]):
    row = dflast.iloc[i, 2674] 
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
  
## BEGIN EXPERIMENTS

# Find the feature cols
cols1 = []
for i in range(0, cols - 1, 1):
    cols1.append(i)
    
## Run accuracy on original dataset
acc1 = runClassifier(df100, 'SVM', 'f1')


# ## Instantiate the list for the accuracies
# augAcc = [0] * (len(randUnit) - 1)
# augAcc.insert(0, acc1)

# augAcc2 = [0] * (len(pmUnit) - 1)
# augAcc2.insert(0, acc1)

# augAcc3 = [0] * (len(gausNoise) - 1)
# augAcc3.insert(0, acc1)

# augAcc4 = [0] * (len(modGausNoise) - 1)
# augAcc4.insert(0, acc1)

augAcc = [0] * (len(rowIter) - 1)
augAcc.insert(0, acc1)

augAcc2 = [0] * (len(rowIter) - 1)
augAcc2.insert(0, acc1)

augAcc4 = [0] * (len(rowIter) - 1)
augAcc4.insert(0, acc1)

augAcc3 = [0] * (len(rowIter) - 1)
augAcc3.insert(0, acc1)


## Experiment for finding the optimal percent of rows to augment
# finAcc = []  

# for data in data:
#     dftest = pd.concat([dflast.iloc[:data, :data], diab2[:data]], axis = 1, ignore_index = True)
#     rows = np.arange(0, data, 5)
#     acc1 = runClassifier(dftest, 'SVM', 'f1')
#     augAcc3 = [0] * (len(rows) - 1)
#     augAcc3.insert(0, acc1)
    
#     cols1 = []
#     for i in range(0, data - 1, 1):
#         cols1.append(i)
        
    
#     for j in range(ITERATIONS):
#         for i in range(1, len(rows)):
#             counter += 1
        
#             print(str(counter / (ITERATIONS * (len(rows)-1))*100)[:4] + '%')
#             x = applyAugmentationMethod(dftest, 'pmOne', rows[i], 20, unit = 1)
#             log = betterLogReg(dataset = x, feature_cols = cols1, target = data, split = data)
#             acc = runClassifier(log, 'SVM', 'f1')
#             augAcc3[i] += acc
            
        
#     augAcc3 = np.asarray(augAcc3)
#     augAcc3[1:] /= ITERATIONS
#     augAcc3 *= 100
    
#     print(max(augAcc3))
#     print(data)
    
#     augAcc3 = augAcc3.tolist()
#     finAcc.append((rows[augAcc3.index(max(augAcc3))]/ data )* 100)

#     fig, ax = plt.subplots()
#     ax.plot(rows, augAcc3, marker = 'o', linewidth = 3.0, color = 'white')
#     ax.set_title('Accuracy vs Rows Augmented')
#     ax.set_xlabel('Unit')
#     ax.set_ylabel('Accuracy')
#     ax.set_facecolor('magenta')
#     plt.show()
  
## Experiment for accuracy and pmOne perturbation  
for j in range(ITERATIONS):
    for i in range(1, len(rowIter)):
        counter += 1
    
        print(str(counter / (ITERATIONS * (len(rowIter)-1))*100)[:4] + '%')
        x = applyAugmentationMethod(df100, 'pmOne', rowIter[i], cols, unit = 0.75)
        log = betterLogReg(dataset = x, feature_cols = cols1, target = cols, split = rows)
        acc = runClassifier(log, 'SVM', 'f1')
        augAcc2[i] += acc
        
    
augAcc2 = np.asarray(augAcc2)
augAcc2[1:] /= ITERATIONS
augAcc2 *= 100

# ## Experiment for accuracy and randUnit perturbation
# for k in range(ITERATIONS):
#     for l in range(1, len(rowIter)):
#         counter += 1
#         print(str(counter / (ITERATIONS * (len(rowIter)-1))*100)[:4] + '%')
#         # x = applyAugmentationMethod(dftest, 'randSwap', 50, value)
#         x = betterRandSwap(df100, rowIter[l], 25)
#         log = betterLogReg(dataset = x, feature_cols = cols1, target = cols, split = rows)
#         acc = runClassifier(log, 'SVM', 'f1')
#         augAcc[l] += acc
    
# augAcc = np.asarray(augAcc)
# augAcc[1:] /= ITERATIONS
# augAcc *= 100

# ## Experiment for accuracy and randUnit perturbation
# for m in range(ITERATIONS):
#     for n in range(1, len(gausNoise)):
#         counter += 1
#         print(str(counter / (ITERATIONS * (len(gausNoise)-1))*100)[:4] + '%')
#         # x = applyAugmentationMethod(dftest, 'randSwap', 50, value)
#         x = betterGausNoise(df100, 200, 100, noise = gausNoise[n])
#         log = betterLogReg(dataset = x, feature_cols = cols1, target = cols, split = rows)
#         acc = runClassifier(log, 'SVM', 'f1')
#         augAcc3[n] += acc
    
# augAcc3 = np.asarray(augAcc3)
# augAcc3[1:] /= ITERATIONS
# augAcc3 *= 100

# ## Experiment for accuracy and randUnit perturbation
# for o in range(ITERATIONS):
#     for p in range(1, len(modGausNoise)):
#         counter += 1
#         print(str(counter / (ITERATIONS * (len(modGausNoise)-1))*100)[:4] + '%')
#         # x = applyAugmentationMethod(dftest, 'randSwap', 50, value)
#         x = modifiedGausNoise(df100, 200, modGausNoise[p])
#         log = betterLogReg(dataset = x, feature_cols = cols1, target = cols, split = rows)
#         acc = runClassifier(log, 'SVM', 'f1')
#         augAcc4[p] += acc
    
# augAcc4 = np.asarray(augAcc4)
# augAcc4[1:] /= ITERATIONS
# augAcc4 *= 100

##Graphs of experiments vs accuracy        
fig, ax = plt.subplots()
ax.plot(rowIter, augAcc2, marker = 'o', linewidth = 3.0, color = 'blue')
ax.set_title('Accuracy vs pmOne')
ax.set_xlabel('Unit')
ax.set_ylabel('Accuracy')
ax.set_facecolor('white')
plt.show()

plt.plot(randUnit, augAcc, marker = 'o', linewidth = 3.0, color = 'blue')
plt.title('Accuracy vs randUnit')
plt.xlabel('Columns')
plt.ylabel('Accuracy')
plt.show()

plt.plot(gausNoise, augAcc3, marker = 'o', linewidth = 3.0, color = 'blue')
plt.title('Accuracy vs gausNoise')  
plt.xlabel('Noise')
plt.ylabel('Accuracy')
plt.show()


plt.plot(randUnit, augAcc4, marker = 'o', linewidth = 3.0, color = 'blue')
plt.title('Accuracy vs modGausNoise')
plt.xlabel('Noise')
plt.ylabel('Accuracy')
plt.show()




# list1 = []
# for i in range(dffinal2.shape[1]):
#     # if (df.iloc[:, i].dtypes == "str"):
#     #     dfdrop2 = dfdrop.drop(axis = 1, labels = dfdrop.columns[i])
#     list1.append(dffinal2.iloc[:, i].dtypes)
    

    
    
    
# list2 = []
# for value in list1:
#     if(value ==  )
        
# dfreal = dfdrop2.astype(float)

# plt.scatter(x = df["WRKDHRSWK2"], y = df['CIG30USE'])

# cols = ['IRALCFY','WRKDHRSWK2']

# # Function that cleans continuous numerical data counting by year
# def cont_clean_data(x):

#     # Survey codes for "Bad Data", Don't Know, Skip, Refused, or Blank
#     if (x == -9) or (x == 985) or (x == 989) or ((x >= 994) and (x < 1000)):
#         return np.nan

#     # Codes for "Have never done..." or "Have not done in the past X days"
#     # Equivalent to 0 for numbered questions
#     if ((x == 991) or (x == 993)):
#         return 0

#     # Ignore value if conditions don't match
#     return x 

# df = df.applymap(cont_clean_data)


# # Function that cleans all other special data codes
# def ord_clean_data(x):

#     # Survey codes for "Bad Data", Don't Know, Skip, Refused, or Blank
#     if ((x == -9) |
#     ((x >= 94) & (x < 100)) |
#     (x == 85) |
#     (x == 89)):
#         return np.nan

#     # Codes for "Have never done..." or "Have not done in the past X days"
#     # Equivalent to 0 for numbered questions
#     if ((x == 91) |
#     (x == 93)):
#         return 0

#     # Ignore value if conditions don't match
#     return x 

# # Changes for wrkdhrsw2
# df[(df['IRWRKSTAT'] == 3) | (df['IRWRKSTAT'] == 4), "WRKDHRSWK2"] = 0
# df.loc[df['WRKDHRSWK2'] == 61, "WRKDHRSWK2"] = np.nan

# # Changes for binary categorical variables
# df.loc[(df['COCEVER'] == 2), "COCEVER"] = 0
# df.loc[(df['CRKEVR'] == 2), "CRKEVR"] = 0
# df.loc[(df['IRSEX'] == 2), "IRSEX"] = 0

# Apply clean_data functions
# df[cont_cols] = df[cont_cols].applymap(cont_clean_data)
# df[ord_cols] = df[ord_cols].applymap(ord_clean_data)



import seaborn as sns
# # Function for easily plotting sns barplots on a grid
# def plot_bar(data, grid, x, y, xlabel, ylabel, title, xticklabels, rotation=0):
#     ax = fig.add_subplot(grid[0], grid[1], grid[2])
#     sns.barplot(data=data, x=x, y=y, 
#     estimator=(lambda x: sum(x)/len(x)), ax=ax).set_title(title)
#     ax.set(xlabel=xlabel, ylabel=ylabel)
#     ax.set_xticklabels(xticklabels, rotation=rotation)

# # Set figure parameters
# plt.rcParams['figure.figsize'] = [16, 12]
# plt.rcParams['figure.subplot.wspace'] = 0.3
# plt.rcParams['figure.subplot.hspace'] = 0.7
# fig = plt.figure()

# # Call plot_bar to plot bar graphs for various variables
# plot_bar(df,[2, 2, 1], 'DIABETEVR', 'IRALCFY', 'Diabetes Diagnosis', 'Days Consumed Alcohol', 
# "Average # Days Consumed Alcohol\nin a Year vs Diabetes Diagnosis", ["Has Diabetes", "Does Not Have Diabetes"])

# # plot_bar(df,[2, 2, 2], 'DIABETEVR', 'WRKDHRSWK2', 'Diabetes Diagnosis', 'Hours Worked', 
# # "Average # Hours Worked in a\nWeek vs Diabetes Diagnosis", ["Has Diabetes", "Does Not Have Diabetes"])

# plot_bar(df,[2, 2, 2], 'DIABETEVR', 'IRPINC3', 'Diabetes Diagnosis', 'Income Range', 
# "Average Income Range\n vs Diabetes Diagnosis", ["Has Diabetes", "Does Not Have Diabetes"])

# plot_bar(df,[2, 2, 3], 'DIABETEVR', 'IRSEX', 'Diabetes Diagnosis', 'Sex', 
# "Sex vs Diabetes Diagnosis", ["Has Diabetes", "Does Not Have Diabetes"])

# plot_bar(df,[2, 2, 4], 'DIABETEVR', 'CATAG3', 'Diabetes Diagnosis', 'Age Group', 
# "Average Age Group\n vs Diabetes Diagnosis", ["Has Diabetes", "Does Not Have Diabetes"])

# Stolen from Jeff
# Exploration graphs
# fig, ax = plt.subplots(2, sharey=False)
# g = sns.histplot(df, x='IRSEX', hue='DIABETEVR', multiple='dodge', ax=ax[0]).set(title='Sex', xlabel = 'Sex')

# sns.histplot(df, hue='DIABETEVR', x='CATAG3', multiple='dodge', ax=ax[1]).set(title='Age Category', xlabel = 'Age')

# plt.tight_layout()
# plt.show()

# plot_bar(df,[2, 2, 2], 'ireduhighst2', 'coccrkever', 'Highest Completed Education', 'Proportion of People Who\nHave Used Crk/Cocaine', 
# "Proportion of People who have\nUsed Crk/Coc by Highest Completed Education",
# ["5th or less", "6th", "7th", "8th", "9th", "10th", "11th/12th,\nno diploma,", 
# "High school\ndiploma/GED", "Some college,\nno degree", "Associate's Deg.", "College Grad\nor Higher"], 90)
# print(df.corr())

# X = df.drop(df["DIABETEVR"])   #Feature Matrix
# y = df["DIABETEVR"]          #Target Variable
# df.head()

# plt.figure(figsize=(12,10))
# cor = df.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()


'''
1
0.43076923076923074
1
3.8605604921394394
1
3.396438574867567
1
3.458507918046653
1
4.540320205453061
1
3.918848083721501
1
3.942425757955234
1
3.9263311040159943
1
3.48838335932527
1
3.6617554338808693
1
3.9731962543171715
1
3.876564948851012
1
3.999121178623742
1
3.754598934648023
1
4.039454057813957
1
3.7624889586937407
1
3.121816864818181
1
4.130058916403672
1
4.000017407472559
1
3.990398014560273
1
3.714097888676106
1
3.745231064473071
1
4.251392538051706
1
3.7190423163195443
1
3.7170203197818004
1
3.6686820435747967
1
3.8109696372445816
1
3.627474494522801
1
3.9062185403911265
1
4.3787433259346855
1
3.9605899243240152
1
4.019139642964211
1
4.379312461099257
1
3.8247513561127913
1
4.096800925753518
1
3.9357212231387595
1
3.432306860964417
1
3.146662980887407
1
3.3195263262701076
1
3.1814227862246156
1
3.187330316445207
1
3.430266887483275
1
3.409710963736872
1
3.3736742424242423
1
3.4221866542985144
1
3.3233227491391477
1
3.152638310312639
1
3.419407716875121
1
3.2270671327452076
1
3.1547502181224756
1
3.2095498906620175
1
3.559451545404854
1
4.0968376832458056
1
3.639028720149713
'''