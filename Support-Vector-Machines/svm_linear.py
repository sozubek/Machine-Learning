'''
We implement support vector machine with linear kernel to randomly generated data.
To find the penalty parameter C of the error term, we do stratified 10-fold cross-validation.
We find the optimal value of C according to the cross-validation error. We obtain 90% 
success rate on the training set and 90.18% and success rate on the cross-validation set.  
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV


def generateData():
	# create a random set of 550 points that belong to a total of 4 classes
	n_samples_1, n_samples_2, n_samples_3, n_samples_4 = 200,150,100,100	
	
	gen = np.random.RandomState(0)
	X = np.r_[gen.randn(n_samples_1, 2), gen.randn(n_samples_2, 2) + [2, 3], 
	gen.randn(n_samples_3, 2) + [3.5, 0], gen.randn(n_samples_4, 2) + [-1, 4]]
	
	y = [0] * (n_samples_1) + [1] * (n_samples_2) + [2] * (n_samples_3)+ [3] * (n_samples_4)
	
	return X,y
	
	
def plotData(X,y):
	plt.scatter(X[:,0], X[:,1], c=y, cmap = plt.cm.rainbow, s=25)


def plotDecisionBoundary(X, y, model, C):	
	# create a grid of values
	x1_min, x2_min = np.min(X, axis=0)
	x1_max, x2_max = np.max(X, axis=0)
	h = 400				# partition size of the grid
	xx1, xx2 = np.meshgrid(np.linspace(x1_min-1, x1_max+1, h), np.linspace(x2_min-1, x2_max+1, h))
	
	# evaluate the model on the grid
	Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
	Z = Z.reshape(xx1.shape)
	
	# plot the decision boundary according to the model's predictions on the grid
	plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired, alpha=0.5)
	plt.axis([-2.5,2.5,-2.5,2.5])
	plt.title('SVM with Linear Kernel (C = %.2f)' %(C))
	
	plotData(X,y)		
		

def gridSearchCV(X, y, kernel):
	# set the values for the parameter to be examined
	values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
	parameters = dict(C=values)
	
	# set the K-Folds cross-validation iterator
	cv = StratifiedKFold(y=y, n_folds=10)
	
	# search for the best value of C over the possible values
	clf = GridSearchCV(SVC(kernel=kernel), param_grid=parameters, cv=cv)
	return clf
#	return clf.best_params_, clf.score(X,y), clf.best_score_


	
def main():
	X,y = generateData()	 	
 	
 	# normalize features by removing the mean and scaling to unit variance
 	scaler = preprocessing.StandardScaler()
 	X_norm = scaler.fit_transform(X)
 	
 	# select the kernel
	kernel = 'linear'
	
	# find the best classifier and fit to the model
	clf = gridSearchCV(X_norm, y, kernel)
	clf.fit(X_norm, y)
				
 	# plot the decision boundary
 	C = clf.best_params_['C'] 
 	plotDecisionBoundary(X_norm, y, clf, C)
 	
 	# report the success rate
 	print 'The success rate on the training set: %.2f%%' %(clf.score(X_norm,y)*100)
 	print 'The success rate on the cross-validation set: %.2f%%' %(clf.best_score_*100)
 	
	plt.show()
 	
 
if __name__ == '__main__':
	main()
