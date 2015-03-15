'''
We implement logistic regression to non-linear data using BFGS method. To account for 
the non-linearity, we add polynomial features. We also use regularization to prevent
overfitting. We plot the decision boundary and calculate training accuracy (99.58%).
The data is the FLAME data set and originates from the following article:
Fu, L. and E. Medico, FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data. 
BMC bioinformatics, 2007. 8(1): p. 3.  
The data is available at  http://cs.joensuu.fi/sipu/datasets/ .
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 
import scipy as sp


def loadData(filename):
	data = np.genfromtxt(filename)
	X = data[:,:2]
	y = data[:,-1] - 1 	
	return X,y


def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))


def costFunctionReg(theta, *args):
	X, y, Lambda = args[0], args[1], args[2]
	m = len(y)

	# A is an auxiliary matrix to get rid of theta[0] during regularization
	A = np.eye(len(theta)) 
	A[0,0] = 0
	
	cost = (1.0 / m) * (np.dot(-y.T, np.log(sigmoid(np.dot(X, theta)))) - np.dot(1-y.T, np.log(1 - sigmoid(np.dot(X, theta)))))\
	      + Lambda / (2 * m) * np.dot(np.dot(A, theta).T, np.dot(A, theta))
	return cost
	
	
def costGradientReg(theta, *args):
	X, y, Lambda = args[0], args[1], args[2]
	m = len(y)

	# A is an auxiliary matrix to get rid of theta[0] during regularization
	A = np.eye(len(theta)) 
	A[0,0] = 0
	
	grad = (1.0 / m) * np.dot(X.T, sigmoid(np.dot(X, theta)) - y)\
	       + (Lambda / m) * np.dot(A, theta)
	return grad


def mapFeature(X,n):
	# adds polynomial features up to and including the nth power
	poly = preprocessing.PolynomialFeatures(n)
	return poly.fit_transform(X)	


def predict(theta, X):	
	# predict 0 if the logit value < 0.5, predict 1 if it is >= 0.5
	return np.floor(sigmoid(np.dot(X,theta))+ 0.5)


def plotData(X,y):
	colors = ['y', 'r']
	labels = ['Type 1', 'Type 2']	
	for i in range(2):
		plt.scatter(X[y==i][:,0], X[y==i][:,1], color = colors[i], marker = 'o', s = 20, label = labels[i]) 
	plt.legend()	


def plotDecisionBoundary(theta, X, y, Lambda, degree):
	# find the boundary values for the features
	x1min, x2min = X[:,[1,2]].min(axis=0)
	x1max, x2max = X[:,[1,2]].max(axis=0)

	# make predictions on a grid bordered by the boundary values
	xx1, xx2 = np.meshgrid(np.linspace(x1min, x1max, 800), np.linspace(x2min, x2max, 800))
	Z = mapFeature(np.c_[xx1.ravel(), xx2.ravel()], degree)
	p = predict(theta, Z)
	p = p.reshape(xx1.shape)
	
	# plot the decision boundary according to the predictions on the grid
	plt.figure()
	plotData(X[:,[1,2]], y)
	plt.contour(xx1, xx2, p, levels = [0], colors='g', linewidths=2)
	plt.title('Nonlinear Logistic Regression (Lambda:%.2f)' %Lambda)
	
	
def normalize(X):
	# normalize the features matrix with mean=0 and std=1
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	X_norm = (X - mean) / std
	return X_norm, mean, std	



def main():
	X,y = loadData('flame.txt')

	# normalize X
	X, mean, std = normalize(X)

	# add polynomial features
	degree = 3
	X = mapFeature(X,degree)

	# set parameters
	Lambda = 1.0						# regularization parameter
	initial_theta = np.zeros((X.shape[1]))
	args = (X, y, Lambda)

	# find the optimal theta using bfgs method
	result = sp.optimize.fmin_bfgs(costFunctionReg, initial_theta, 
		fprime = costGradientReg, args=args, maxiter=1000, full_output=True)	
	theta = result[0]
	print 'The optimal theta is:', result[0]
	print 'The mininum value for the cost function is %f' %(result[1])
	
	plotDecisionBoundary(theta, X, y, Lambda, degree)
	
	# find the training accuracy
	p = predict(theta, X)
	print 'Training accuracy: %.2f%%' %(np.mean(p==y) * 100)
	
	plt.show()
	


if __name__ == '__main__':
	main()	
	
	
