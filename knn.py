#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:06:39 2022

@author: ngelean
"""

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from math import sqrt
import time



import warnings
warnings.filterwarnings('ignore')


# KNN_Start
def loadData(filename):
    # Load data from file into X
    X = []
    count = 0
    
    text_file = open(filename, "r")
    lines = text_file.readlines()
        
    for line in lines:
        X.append([])
        words = line.split(",")
        # Convert values of the first attribute into float
        for word in words:
            if (word=='M'):
                word = 0.333
            if (word=='F'):
                word = 0.666
            if (word=='I'):
                word = 1
            X[count].append(float(word))
        count += 1
    
    return np.asarray(X)


def testNorm(X_norm):
    xMerged = np.copy(X_norm[0])
    # Merge datasets
    for i in range(len(X_norm)-1):
        xMerged = np.concatenate((xMerged,X_norm[i+1]))
    print(np.mean(xMerged,axis=0))
    print(np.sum(xMerged,axis=0))


# This is an example the main of KNN with train-and-test + Euclidean
def knnMain(filename,percentTrain,k):
 
    # Data load
    X = loadData(filename)
    # Normalization
    X_norm = dataNorm(X)
    # Data split: train-and-test
    X_split = splitTT(X_norm,percentTrain)
    # KNN: Euclidean
    accuracy = knn(X_split[0],X_split[1],k)
    
    return accuracy


# Normalisation 
def dataNorm(X):
    X_norm = np.empty(X.shape)
    # 8 normalized input attributes plus 1 non-normalized output attribute
    for x in range(X.shape[1] -1):
        X_norm[:,x]  =(X[:,x] - np.min(X[:,x]))/(np.max(X[:,x])-np.min(X[:,x]))
    X_norm[:,-1] = X[:,-1]
    return X_norm


# Splitting (TT)
def splitTT(X_norm, percentTrain):
    X = []
    I = X.shape[0]
    # shuffle the data with numpy.random.shuffle() before splitting the dataset
    np.random.shuffle(X_norm)
    # X_train for X_split
    X_train = X_norm[ :round(I*percentTrain),:]
    X_test = X_norm[round(I*percentTrain):, ]

    X_split = [X_train, X_test]
    return X_split

# splitCV() , that takes in the normalized dataset X_norm and the value k
def splitCV(X_norm, k):
    np.random.shuffle(X_norm)
    return np.array_split(X_norm,k)

# Euclidean Distance
def euclideanDistance(rowA, rowB):
    distance = 0.0
    for x in range(len(rowA)-1):
        distance += (rowA[x] - rowB[x])**2
    return sqrt(distance)

# Getting neighbour
def get_neighbours(training_data, test_row, num_neighbors):
    distances = list()
    for train_row in training_data: 
        distance = euclideanDistance(test_row, train_row)
        distances.append((train_row, distance))
    distances.sort(key=lambda tup: tup[1])
    neighbours = list()
    for n in range(num_neighbors):
        neighbours.append(distances[n][0])
    return neighbours
 
# Making classifiation prediction with the neighbours above
def predictClassification(training_data, test_row, num_neighbors):
    neighbours = get_neighbours(training_data, test_row, num_neighbors)
    outputValues = [row[-1] for row in neighbours]
    # Getting the most represented class among the neighbours
    prediction = max(set(outputValues), key=outputValues.count)
    return prediction
    
    
# The KNN() function should take in the training dataset X_train , testing dataset X_test , 
# and the number of nearest neighbors, K , and returns the accuracy score as a result of the classification.
# Get accruacy % 
def accuracyMetric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def KNN(X_train, X_test, num_neighbors):
    predictions = []
    actualVal = [a[-1] for a in X_test]
    for row in X_test:
        out = predictClassification(X_train, row, num_neighbors)
        predictions.append(out)
    return predictions, actualVal
 
def knn(train, test, num_neighbors):
    predictions, actual_val=KNN(train, test, num_neighbors)
    accuracy = accuracyMetric(actual_val, predictions)
    return accuracy

# Get knnMain for TT
def knnMain_TT(filename,percentTrain,k):
    # Data load
    X = loadData(filename)
    # Normalization
    X_norm = dataNorm(X)
    # Data split: train-and-test
    X_split = splitTT(X_norm, percentTrain)
    # KNN: Euclidean
    accuracy = knn(X_split[0],X_split[1],k)
    
    return accuracy

# Get knnMain for CV
def knnMain_CV(filename,n_folds,k):
    # Data load
    X = loadData(filename)
    # Normalization
    X_norm = dataNorm(X)
    # Data split: train-and-test
    newSplit = splitCV(X_norm, n_folds)
    # KNN: Euclidean
#     newSplit = np.random.sample(X_split, len(X_split))
    test = newSplit[n_folds-1]
    training_set = newSplit[:n_folds-1]
    for train in training_set:
        accuracy = knn(train, test, k)
        return accuracy

# To test
# print()
# print("15, 20")
# t0 = time.process_time()
# result_accuracy0=knnMain_CV('abalone.data', 15, 20)
# elapsed_time0 = time.process_time() - t0

# print(result_accuracy0)
# print(elapsed_time0)


