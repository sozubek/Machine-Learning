We implement KNN algorithm on the Wine Data Set from UCI Machine Learning Repository. 
Firstly we normalize the data and perform principal component analysis so that
90% of the variance is retained. We then do 10-fold stratified cross-validation to get
the best value for k. We run KNN algorithm on the test data with this value of k and 
obtain a mean accuracy of 97.22%.
The Wine Data Set contains the results of a chemical analysis of wines grown in the same region 
in Italy but derived from three different cultivars. The data contains 13 variables and 178 instances. 
It is available at https://archive.ics.uci.edu/ml/datasets/Wine .
