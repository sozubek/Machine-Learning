'''
We implement random forest algorithm to the Breast Cancer Wisconsin Data Set.
Firstly we split the data into training and test sets with the ratio 80-20. 
We look for the best values for the number of trees in the forest and the number
of features to consider at each split by grid search. For each possible combination
of values we do a 10-fold stratified cross-validation on the training set. 
Finally we calculate the accuracy on the test set (99.27%) and the training set (100%).
We also report the most important features in the data.
The data set is available at https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original) .
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV


def loadData(filename):
	data = np.genfromtxt(filename, delimiter=',')
	X = data[:,1:-1]
	y = data[:,-1]
	return X,y	


def gridSearchCV(X, y):
	# set the possible values for the parameters to be examined
	num_trees = [10, 20, 30, 40, 50]
	num_features = [2, 3, 4, 5]
	parameters = dict(n_estimators=num_trees, max_features=num_features)
	
	# set the K-Folds cross-validation iterator
	cv = cross_validation.StratifiedKFold(y=y, n_folds=10)
	
	# search for the best value of num_trees in the forest and 
	# num_features to consider at each split 
	grid = GridSearchCV(RandomForestClassifier() , param_grid=parameters, cv=cv)
	grid.fit(X, y)
	return grid.best_params_
	
		
def main():
	X, y = loadData('breast-cancer-wisconsin.data')	
	feature_names = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 
					'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses']
	np.random.seed(1)
	
	# split the data into training and test sets
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

	# find the best classifier from the grid search
	params = gridSearchCV(X_train, y_train)
	n_estimators = params['n_estimators'] 
 	max_features = params['max_features']
	
	# fit the classifier to the training data with the best parameters	
	clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features) 
	clf.fit(X_train, y_train)
	
	# report the training and test scores
	print 'Accuracy on the test set: %.2f%%' %(clf.score(X_test,y_test)*100) 
	print 'Accuracy on the training set: %.2f%%\n' %(clf.score(X_train,y_train)*100) 
	
	# report the most important features
	feature_importances = clf.feature_importances_
	sorted_feature_importances = np.argsort(feature_importances)[::-1]
	j = clf.transform(X).shape[1]
	print 'The most important %d features are:' %j
	for i ,index in enumerate(sorted_feature_importances[:j]):
		print '%d) %s (%.2f%%)' %(i+1, feature_names[index], feature_importances[index]*100)


if __name__ == '__main__':
	main()	
