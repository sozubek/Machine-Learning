'''
We implement KNN algorithm on the Wine Data Set from UCI Machine Learning Repository. 
Firstly we normalize the data and perform principal component analysis so that
90% of variance is retained. We then do 10-fold stratified cross-validation to get
the best value for k. We run KNN algorithm on the test data with this value of k and 
obtain a mean accuracy of 97.22%.
The data contains the results of a chemical analysis of wines grown in the same region 
in Italy but derived from three different cultivars. The data contains 13 variables and  
is available at https://archive.ics.uci.edu/ml/datasets/Wine .
'''

import numpy as np
from scipy import linalg
from sklearn import cross_validation


def loadData(filename):
	data = np.genfromtxt(filename, delimiter=',')
	X = data[:,1:]
	y = (data[:,0]-1).astype(int) 	
	return X,y
	
	
def featureNormalize(X):
	# normalize the features with mean=0 and std=1
	mu = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	X_norm = (X - mu) / std
	return X_norm, mu, std


def pca(X):
	# compute the covariance matrix
	sigma = np.cov(X, rowvar=0)	
	
	# compute the singular value decomposition of the covariance matrix
	U, S, Vh = linalg.svd(sigma)	
	
	return U, S

	
def projectData(X, U, num_comp):
	# project the data to the subspace spanned by num_comp principal components
	U_reduce = U[:, range(num_comp)]
	return np.dot(X, U_reduce)
	
	
def runKNN(X, y, X_test, k):
	m = X_test.shape[0]
	predictions = np.array([-1]*m)
	for i in range(m):
		# find the distance of a sample to each point in the training set  
		distances = np.linalg.norm(X - X_test[i], axis=1)
		
		# get the labels of the k closest points
		k_neighbour_labels = y[np.argsort(distances)[:k]]
		
		# make a prediction by the majority vote among the k closest points
		predictions[i] = np.argmax(np.bincount(k_neighbour_labels))
	return predictions
	
	
def main():
	X, y = loadData('wine.data')
	
	# split the data into training and test sets
	np.random.seed(0)
	X, X_test, y, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
	
	# standardize the features of the training set with mean=0 and std=1
	X_norm, mu, std = featureNormalize(X)
	
	# standardize the features of the test set using the mean and std of the training set
	X_test_norm = (X_test - mu) / std 
		
	# run PCA and get the array of eigenvectors	and the singular values
	U, S = pca(X_norm)
	
	# find the minimum number of principal components so that 90% of variance is retained
	num_comp = 0				# number of principal components
	for i in range(len(S)):
		if (np.sum(S[:i+1]) / np.sum(S)) >= 0.90:
			num_comp = i
			break	

	# project the training and test data to the subspace spanned by num_comp principal components
	Z = projectData(X_norm, U, num_comp)	
	Z_test = projectData(X_test_norm, U, num_comp)
	
	# find the best value for k by 10-fold stratified cross-validation
	k_values = [3,5,8,10,13,15]
	mean_accuracies = np.array([])
	for k in k_values:
		skf = cross_validation.StratifiedKFold(y, n_folds=10)
		accuracy_list = np.array([])
		for train_index, cv_index in skf:
			Z_train, Z_cv = Z[train_index], Z[cv_index]
			y_train, y_cv = y[train_index], y[cv_index]
			predictions = runKNN(Z_train, y_train, Z_cv, k)
			accuracy = np.mean(predictions == y_cv)
			accuracy_list = np.append(accuracy_list, accuracy)
		mean_accuracies = np.append(mean_accuracies, np.mean(accuracy_list))	
	best_k = k_values[np.argmax(mean_accuracies)]		
			 	
	# find the predictions on the test set using the best value of k, report the accuracy
	predictions = runKNN(Z, y, Z_test, best_k)
	accuracy = np.mean(predictions == y_test) * 100
	print 'The mean accuracy of KNN on the test set (K=%d): %.2f%%' %(best_k, accuracy)		
	

if __name__ == '__main__':
	main()	
