# KNN using Abalone Dataset


## Learning Objectives:
- To familiarize with the implementation of a machine learning algorithm from scratch, without the usage of any machine learning API.
- To apply KNN algorithm to classify the age of abalone using the Abalone dataset.
- To familiarize with evaluating the performance of a machine learning algorithm


## Dataset
There are 4,177 data observations in the dataset with 8 input attributes and 1 output variable. The input attributes are as follows:
  1. Sex [Male (M), Female (F), or Infant (I)]
  2. Length
  3. Diameter
  4. Height
  5. Whole weight
  6. Shucked weight
  7. Viscera weight
  8. Shell weight
  9. Rings (output)
  
  
  
## Tasks
1. Use the function `loadData()` to load data from file. The command `X = loadData(‘abalone.data’)` returns an array of size .
In this function, the values of the first attribute have been converted into floats:
  - M: 0.333
  - F: 0.666
  - I: 1.000
  
  
2. Normalize the dataset. You are to normalize the 8 input attributes by writing a function, `dataNorm()`. 
The normalization equation is given as: `(data-min)/(maxmin)`


3. Split the dataset into training and testing set by:
  (i) Using the train-and-test split method.
  (ii) Using the -fold cross-validation method. Note that the k -value here is different from the  `K-value` in the KNN algorithm. Set the `k` value to 5, 10 and 15 respectively.
  
  
4. Implement the KNN algorithm by writing a function, `KNN()` . You can use the Euclidean distance as the similarity measure for any two samples.


5. Use the `classification_report()` function provided by the scikit-learn library to construct aclassification report for the 5-fold cross validation with K = 15


