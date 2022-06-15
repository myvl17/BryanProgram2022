# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:08:15 2022

@author: cdiet
"""
# From DataCamp

# import pandas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def LogRegFake(dataset, name, feature_cols, target):
    # read in the file
    txt = "txt"
    csv = "csv"
    if csv in dataset:
        name = pd.read_csv(dataset)
        
    elif txt in dataset:
        name = pd.read_table(dataset, delimiter = " ", header = None)
        name.rename(columns = {150: target}, inplace = True)
    
    # Feature variables
    X = name[feature_cols]
    
    # Target variable
    y = name[target]
    
    # Split both x and y into training and testing sets
    
    # Import train_test_split
    from sklearn.model_selection import train_test_split
    
    # Splitting the sets
    # Test_size indicates the ratio of testing to training data ie 0.25:0.75
    # Random_state indicates to select data randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 
    
    
    # import the class
    from sklearn.linear_model import LogisticRegression
    
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(max_iter = 10000)
    
    # fit the model with data
    logreg.fit(X_train,y_train)
    
    # create the prediction
    y_pred= logreg.predict(X_test)
    
    # Confusion matrix
    # import the metrics class
    from sklearn import metrics
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix
    
    # Visualize the confusion matrix using a heat map
    class_names = [0, 1] # name  of classes
    
    # initialize the plot
    fig, ax = plt.subplots()
    
    # Plot the tick marks
    tick_marks = np.arange(len(class_names))
    
    # Both take ticks and labels
    plt.xticks(ticks = tick_marks, labels = class_names)
    plt.yticks(ticks = tick_marks, labels = class_names)
    
    # create heatmap with labels, specify colors in cmap
    # Annot writes the numbers in the cells
    # fmt set to g is for integers, d is decimal, s is string
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="BuPu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    
    # Add labels for readability
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    # On the heat map, the top left is a true positive(8), top right
    # is a false positive(3), bottom left is a false negative(1), and bottom
    # right is a true negative(37)
    
    # Print out an evaluation of the model using accuracy,
    # precision, and recall
    
    # From all positive and negative cases (true and false), this
    # is the amount we predicted correctly
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # 91.8% accuracy, really good
    
    # From all the classes we predicted as positive (true positive and false
    # positive), the amount that are actually positive
    print("Precision:", metrics.precision_score(y_test, y_pred))
    # Model predicted a person has parkinson's and they actually do
    # 92.5% of the time
    
    # From all of the positive classes (true positive and false negative),
    # this is the amount we predicted correctly
    print("Recall:",metrics.recall_score(y_test, y_pred))
    # People who have Parkinson's can be identified 97.4% of the time
    
    # Create new plot for second figure
    fig, ax2 = plt.subplots()
    
    # Finds the probability estimates, takes a vector
    y_pred_proba = logreg.predict_proba(X_test)[::,1]
    
    # Creates a graph of Receiver Operating Characteristic, graph of true
    # positive against false positive
    # Takes in the true values, and predicted values
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc = "+str(auc))
    plt.legend(loc=4)
    plt.show()  
    
    # auc of 1 is a perfect classifier and auc of 0.5 is a 
    # worthless classifier, auc = 0.933
    
# An example for the Logistic Regression, will output confusion matrix and line graph    
array = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                'MDVP:RAP', 'MDVP:PPQ',	'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
                'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
                'spread2', 'D2', 'PPE']
print(LogReg(dataset = "C:/Users/cdiet/Desktop/Parkinson_datset.csv", 
              name = "parkinson", feature_cols = array, target = 'status'))

