We implement random forest algorithm to the Breast Cancer Wisconsin Data Set.
Firstly we split the data into training and test sets with the ratio 80-20. 
We look for the best values for the number of trees in the forest and the number
of features to consider at each split by grid search. For each possible combination
of values we do a 10-fold stratified cross-validation on the training set. 
Finally we calculate the accuracy on the test set (99.27%) and the training set (100%).
We also report the most important features in the data.
The data set is available at https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original) .
