#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-12-11 2:52 PM
# @File    : DataProcessing.py

"""
main code for data analysis
"""
# Importing the libraries
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
sns.set(color_codes=True)
import pandas as pd
import glob
import os
from pandas.plotting import scatter_matrix
# %matplotlib inline
# Importing metrics for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    # read the dataset
    path = r'dataset'  # use your path
    all_files = glob.glob(os.path.join(path, "*.csv"))  # advisable to use os.path.join as this makes concatenation OS independent
    df_from_each_file = (pd.read_csv(f,usecols=range(15)) for f in all_files)
    dataset = pd.concat(df_from_each_file, ignore_index=True)

    # preview of the data set
    print(dataset.head())

    # shape
    print(dataset.shape)

    # more info on the data
    print(dataset.info())

    # descriptions
    print(dataset.describe())

    # class distribution
    print(dataset.groupby('NLOS').size())

    # # box and whisker plots
    # dataset.plot(kind='box', sharex=False, sharey=False)

    # histograms
    dataset.hist(edgecolor='black', linewidth=1)
    #
    # # boxplot on each feature split out by species
    dataset.boxplot(by="NLOS", figsize=(10, 8))
    #
    # # scatter plot matrix
    # scatter_matrix(dataset, figsize=(10, 10))
    # violinplots on petal-length for each species
    for col in dataset.columns:
        if col=='NLOS':
            continue
        else:
            plt.figure()
            sns.violinplot(data=dataset, x="NLOS", y=col)

    # # Using seaborn pairplot to see the bivariate relation between each pair of features
    # sns.pairplot(dataset, hue="NLOS")

    # plt.show()

    # Seperating the data into dependent and independent variables
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # LogisticRegression
    start = time.time()
    # classifier_LR = LogisticRegression(multi_class='auto',solver='newton-cg')
    classifier_LR = LogisticRegression(solver='saga')
    classifier_LR.fit(X_train, y_train)
    y_pred_LR = classifier_LR.predict(X_test)
    # print(classification_report(y_test, y_pred_LR))
    accuracy_LR = accuracy_score(y_test,y_pred_LR)
    time_LR = time.time() - start

    # Decision Tree
    start = time.time()
    classifier_DT = DecisionTreeClassifier()
    classifier_DT.fit(X_train, y_train)
    y_pred_DT = classifier_DT.predict(X_test)
    # print(classification_report(y_test, y_pred_LR))
    accuracy_DT = accuracy_score(y_test,y_pred_DT)
    time_DT = time.time() - start

    # Support Vector Machine
    start = time.time()
    # classifier_SVC = SVC()
    classifier_SVC = SVC(gamma='scale')
    classifier_SVC.fit(X_train, y_train)
    y_pred_SVC = classifier_SVC.predict(X_test)
    # print(classification_report(y_test, y_pred_LR))
    accuracy_SVC = accuracy_score(y_test,y_pred_SVC)
    time_SVC = time.time() - start


    print('The accuracy of Logistic Regression Classifier is', accuracy_LR)
    print('The accuracy of Decision Tree Classifier is', accuracy_DT)
    print('The accuracy of Support Vector Machine Classifier is', accuracy_SVC)

    # calculation time
    print('The time consumption  of Logistic Regression Classifier is', time_LR)
    print('The time consumption  of Decision Tree Classifier is', time_DT)
    print('The time consumption  of Support Vector Machine Classifier is', time_SVC)

    fig, ax = plt.subplots()
    plt.title("Prediction accuracy")
    x_bar = ['Logistic Regression','Decision Tree','Support Vector Machine']
    plt.bar(x_bar,[accuracy_LR,accuracy_DT,accuracy_SVC])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.xlabel("Algorithm")
    plt.ylabel("Prediction accuracy")

    plt.figure()
    plt.title("Time consumption")
    plt.bar(x_bar,[time_LR,time_DT,time_SVC])
    plt.xlabel("Algorithm")
    plt.ylabel("Time consumption [second]")

    plt.show()

