We implement a Naive Bayes classifier using the Nursery Data Set from 
the UCI Machine Learning Repository. The data was derived from a hierarchical decision model 
originally developed to rank applications for nursery schools. It contains 8 categorical
features and 5 classes with 12690 instances. We divide the data into training and test sets with 
the ratio 80-20. We calculate conditional and prior log probabilities with Laplace smoothing. 
For each new sample we predict the class that gives the maximum posterior probability.
We obtain 88.46% accuracy on the test set.
The data set is available at http://archive.ics.uci.edu/ml/datasets/Nursery.
