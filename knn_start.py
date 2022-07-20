import numpy as np


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

