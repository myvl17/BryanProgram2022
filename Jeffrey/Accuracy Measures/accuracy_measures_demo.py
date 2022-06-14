# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:50:46 2022

@author: jeffr
"""


"""
Sci-kit Learn functions for accuracy measure
Inputs: original data, predicted data
Output: "accuracy"
"""

"""

import sklearn.metrics

Accuracy score: Basic accuracy test; Correct Predictions / Total Predictions
sklearn.metrics.accuracy_score(y_true, y_pred)

Absolute mean error: Measure of error between predicted and true value
sklearn.metrics.mean_absolute_error(y_true, y_pred)

Root mean squared error: Squareroot of mean error
sklearn.metrics.mean_squared_error(y_true, y_pred, root=False)

F1 score: Combination of precision and recall into harmonic mean or accuracy (Binary)
sklearn.metrics.f1_score(y_true, y_pred)

# Area under curve score:  (Binary)
sklearn.metrics.roc_auc_score(y_true, y_score)

"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

import sklearn.metrics as skm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import datasets

# Generating random data set
iris = datasets.load_iris()
X = iris.data[:, 0:2]  # we only take the first two features for visualization
y = iris.target

n_features = X.shape[1]

C = 10
kernel = 1.0 * RBF([1.0, 1.0])  # for GPC

# Create different classifiers.
classifiers = {
    "L1 logistic": LogisticRegression(
        C=C, penalty="l1", solver="saga", multi_class="multinomial", max_iter=10000
    ),
    "L2 logistic (Multinomial)": LogisticRegression(
        C=C, penalty="l2", solver="saga", multi_class="multinomial", max_iter=10000
    ),
    "L2 logistic (OvR)": LogisticRegression(
        C=C, penalty="l2", solver="saga", multi_class="ovr", max_iter=10000
    ),
    "Linear SVC": SVC(kernel="linear", C=C, probability=True, random_state=0),
    "GPC": GaussianProcessClassifier(kernel),
}

# Creating scatterplots
n_classifiers = len(classifiers)

plt.figure(figsize=(3 * 2, n_classifiers * 2))
plt.subplots_adjust(bottom=0.2, top=0.95)

xx = np.linspace(3, 9, 100)
yy = np.linspace(1, 5, 100).T
xx, yy = np.meshgrid(xx, yy)
Xfull = np.c_[xx.ravel(), yy.ravel()]


# Determining accuracy for each iteration
for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X, y)
    
    """
    ********************* USEFUL PART *********************************
    Using accuracy, mean absolute error, and rooted mean absolute error to 
    determine accuracy
    
    """

    y_pred = classifier.predict(X)
    accuracy = skm.accuracy_score(y, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    
    ame_accuracy = skm.mean_absolute_error(y, y_pred)
    print("Mean Absolute Error Accuracy for %s: %0.1f%% " % (name, 100-(ame_accuracy * 100)))
    
    rmse_accuracy = skm.mean_squared_error(y, y_pred, squared=False)
    print("Rooted Mean Absolute Error Accuracy for %s: %0.1f%% " % (name, 100-(rmse_accuracy * 100)))
    
    print()
    

    # View probabilities:
    probas = classifier.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    for k in range(n_classes):
        plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
        plt.title("Class %d" % k)
        if k == 0:
            plt.ylabel(name)
        imshow_handle = plt.imshow(
            probas[:, k].reshape((100, 100)), extent=(3, 9, 1, 5), origin="lower"
        )
        plt.xticks(())
        plt.yticks(())
        idx = y_pred == k
        if idx.any():
            plt.scatter(X[idx, 0], X[idx, 1], marker="o", c="w", edgecolor="k")

ax = plt.axes([0.15, 0.04, 0.7, 0.05])
plt.title("Probability")
plt.colorbar(imshow_handle, cax=ax, orientation="horizontal")

plt.show()