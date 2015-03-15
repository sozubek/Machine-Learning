'''
We implement logistic regression to multi class data using one-vs-all method. 
We use regularization to achieve better results. Optimization is done using 
conjugate-gradient method. We plot the decision boundary and calculate training accuracy (92.70%). 
The data comes from the Wine Data Set available at 
https://archive.ics.uci.edu/ml/datasets/Wine .
The data contains the results of a chemical analysis of wines grown in the same region 
in Italy but derived from three different cultivars. Original data contains 13 variables, 
here we use two of them, namely
1) Malic acid;
2) Nonflavanoid phenols;
Using this information we classify three different types of wine.
'''

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def loadData(filename):
	data = np.genfromtxt(filename, delimiter=',')
	X = data[:,[1,7]]
	y = data[:,0]-1 	
	return X,y


def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))


def costFunctionReg(theta, *args):
	X, y, Lambda = args[0], args[1], args[2]
	m = len(y)
	
	# A is an auxiliary matrix to get rid of theta[0] during regularization
	A = np.eye(len(theta)) 
	A[0,0] = 0
		
	cost = 1.0 / m * (-np.dot(y, np.log(sigmoid(np.dot(X, theta)))) - np.dot(1-y, np.log(1-sigmoid(np.dot(X,theta)))))\
	+ Lambda / (2 * m) * np.dot(np.dot(A, theta), np.dot(A, theta))
	
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


def predict(all_thetas, X):
	# calculate the logit value of each sample using the parameters of 
	# each classifier. pick the class for which the corresponding 
	# logistic regression classifier outputs the highest probability
	A = np.dot(all_thetas, X.T)
	prediction = np.argmax(A, axis=0) 
	return prediction
	
	
def plotData(X,y):
	colors = ['y', 'r','b']
	labels = ['Type 1', 'Type 2', 'Type 3']	
	for i in range(3):
		plt.scatter(X[y==i][:,0], X[y==i][:,1], color = colors[i], marker = 'o', s = 20, label = labels[i]) 
	plt.xlabel('Malic Acid')
	plt.ylabel('Nonflavanoid phenols')
	plt.legend()

	
def plotDecisionBoundary(all_thetas, X, y, Lambda):	
	X = X[:,[1,2]]
	plotData(X, y)

	# make predictions on a grid
	x1min, x2min = X.min(axis=0)
	x1max, x2max = X.max(axis=0)
	xx1, xx2 = np.meshgrid(np.linspace(x1min,x1max,500), np.linspace(x2min,x2max,500))
	Z = predict(all_thetas, np.c_[np.ones(len(xx1.ravel())) ,xx1.ravel(), xx2.ravel()])
	Z = Z.reshape(xx1.shape)
	
	# plot the decision boundary according to the predictions on the grid 
	plt.contour(xx1, xx2, Z, linewidths=1.5)
	plt.title('Classification of Wines (Lambda=%.1f)' %(Lambda))	


def main():
	X,y = loadData('wine.data')
	
	# add intercept term to X
	X = np.c_[np.ones(X.shape[0]), X]
	
	# set parameters
	Lambda = 0.3					# regularization parameter
	initial_theta = np.zeros(X.shape[1])

	# train a logistic regression classifier for each class using
	# the conjugate gradient method. Put the optimal parameters of 
	# each classifier into an array
	thetas = []
	for i in range(3):
		args = (X,(y==i),Lambda)
		result = optimize.fmin_cg(costFunctionReg, initial_theta, fprime = costGradientReg, args =args, maxiter=500)
		thetas.append(result)
	all_thetas = np.array([entry[:]for entry in thetas])

	# calculate the training accuracy
	pred = predict(all_thetas, X)
	print 'Training Set Accuracy: %.2f%%' %(np.mean(pred==y)*100)
	
	# plot the decision boundaries 
	plotDecisionBoundary(all_thetas, X, y, Lambda)
	
	plt.show()
	
	
if __name__ == '__main__':
	main()	
