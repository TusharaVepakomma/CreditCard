#importing the modules
#import numpy n-dimensional array
import numpy as np
#import sklearn python machine learning  modules
import sklearn as sk
#import pandas dataframes
import pandas as pd
import unittest
#import matplotlib for plotting
import matplotlib.pyplot as plt
#import datasets  and linear_model from sklearn module
from sklearn import datasets, linear_model
#importing machine learning mopdules Decision Tree and KNN  sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
#import Polynomial features from sklearn module
from sklearn.preprocessing import PolynomialFeatures
#import train_test_split data classification
from sklearn.model_selection import train_test_split
#import ConfusionMatrix from pandas_ml
from pandas_ml import ConfusionMatrix
from django.shortcuts import render

# from models import trees

# Create your views here.
# def Credit(request):
#      return render(request, "index.html")
#
# def ShowPage(request):
#     f = Card()
#     return render(request, "css.html", {'form': f})

# def ShowPage1(request):
#     f = Card1()
#     return render(request, "css.html", {'form1': f})

def index(request):

    # posts = trees.objects.filter().order_by('published_date')

    dataframe = pd.read_csv('D:/UI projects/creditcard/creditcard\creditcard1.csv', low_memory=False)
    #dataframe.sample Returns a random sample of items from an axis of object.
    #The frac keyword argument specifies the fraction of rows to return in the random sample, so frac=1 means return all rows (in random order).
    # If you wish to shuffle your dataframe in-place and reset the index
    #dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    #dataframe.head(n) returns a DataFrame holding the first n rows of dataframe.
    dataframe.head()
    print (dataframe)
    #The loc method gives direct access to the dataframe allowing for assignment to specific locations of the dataframe.
    #here in dataframe class with 1 label is selected for fraud_class
    fraud_class = dataframe.loc[dataframe['Class'] == 1]
    #here in dataframe class with 1 label is selected for non_fraud_class
    non_fraud_class = dataframe.loc[dataframe['Class'] == 0]
    #printing length of fraud_class and non_fraud class
    print("Totally we have ", len(fraud_class), "fraud data class point  and", len(non_fraud_class), "nonfraudulent data class points.")
    #plotting fraudplot for fraud_class
    ax = fraud_class.plot.scatter(x='Amount', y='Class', color='Red', label='Fraud')
    #plotting plot for non_fraud_class with fraud_class
    non_fraud_class.plot.scatter(x='Amount', y='Class', color='Green', label='Normal', ax=ax)
    plt.show()
    print("This Feature what is mentioned is based on the class Distribution.")
    #Let us see the plot zooming only  fraudplot
    bx = fraud_class.plot.scatter(x='Amount', y='Class', color='Orange', label='Fraud')
    #Showing the plot
    plt.show()
    #Again plotting non_fraud_class plot with fraud_class plot
    ax = fraud_class.plot.scatter(x='V15', y='Class', color='Orange', label='Fraud')
    non_fraud_class.plot.scatter(x='V15', y='Class', color='Blue', label='Normal', ax=ax)
    plt.show()
    #Dataframes for  X all the columns except class and y class columns
    X = dataframe.iloc[: , :-1]
    y = dataframe['Class']
    #Finding the length of X and y
    print("X and y sizes, respectively:", len(X), len(y))


    #Splitting the training and Testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=500)
    #Calculating the data
    print("Train and test sizes, respectively:", len(X_train), len(y_train), "|", len(X_test), len(y_test))
    print("Total number of frauds:", len(y.loc[dataframe['Class'] == 1]), len(y.loc[dataframe['Class'] == 1])/len(y))
    print("Number of frauds on y_test:", len(y_test.loc[dataframe['Class'] == 1]), len(y_test.loc[dataframe['Class'] == 1]) / len(y_test))
    print("Number of frauds on y_train:", len(y_train.loc[dataframe['Class'] == 1]), len(y_train.loc[dataframe['Class'] == 1])/len(y_train))
    #Applying Logistic Regression Machine Learning Algorithm
    logistic = linear_model.LogisticRegression(C=1e5)
    #Fitting the Algorithm for X_train and y_train
    logistic.fit(X_train, y_train)
    dt = tree.DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    #Scoring
    print("Using Logistic Regression the Accuracy Score is: ", logistic.score(X_test, y_test))
    print ("Using Decision tree the Accuracy score is ",dt.score(X_test, y_test))
    print ("Using KNearestNeighbour the Accuracy score is ",classifier.score(X_test, y_test))
    y_predicted = np.array(logistic.predict(X_test))
    y_right = np.array(y_test)
    #print y_test
    #The confusion matrix (or error matrix) is one way to summarize the performance of a classifier
    #  for binary classification tasks. This square matrix
    # consists of columns and rows that list the number of instances as absolute or
    # relative "actual class" vs. "predicted class" ratios.
    #Plotting the Confusion matrix for y_right and y_predicted
    confusion_matrix = ConfusionMatrix(y_right, y_predicted)
    print("Confusion matrix:", confusion_matrix)
    confusion_matrix.plot(normalized=True)
    plt.show()
    #printing the stats of Confusion matrix
    confusion_matrix.print_stats()
    # posts = trees.objects.filter(Logistic_Regression, Decison_Tree, K_nearestNeighbour)
    data={'x':len(X),'y':len(y),'fraud':len(fraud_class),'lr':logistic.score(X_test,y_test),'dt':dt.score(X_test,y_test),'kn':classifier.score(X_test,y_test)}

    return render(request, 'index.html',data)
