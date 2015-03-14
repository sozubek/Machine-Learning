'''
We implement logistic regression using Newton-conjugate-gradient method. We plot the
decision boundary, calculate training accuracy and make predictions using the the optimal
variables. The data comes from the Wine Data Set available at 
https://archive.ics.uci.edu/ml/datasets/Wine .
The data contains the results of a chemical analysis of wines grown in the same region 
in Italy but derived from three different cultivars. Original data contains 13 variables, 
here we use two of them, namely
1) Magnesium content;
2) Flavanoid content;
Using this information we classify two different types of wine.
'''

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def loadData(filename):
	data = np.genfromtxt(filename, delimiter=',')
	X = data[:,[1,4]]
	y = data[:,0]-1 	
	X = X[y!=2]
	y = y[y!=2]
	return X,y
	

def sigmoid(X):
	return 1 / (1 + np.exp(-X))


def costFunction(theta, *args):
	X,y = args[0], args[1]
	m = len(y)	
	cost = (1.0 / m) * (np.dot(-y, np.log(sigmoid(np.dot(X, theta))))\
				- np.dot(1-y, np.log(1 - sigmoid(np.dot(X, theta)))))
	return cost

	
def costGradient(theta, *args):
	X,y = args[0], args[1]
	m = len(y)	
 	grad = (1.0 / m) * np.dot(X.T, sigmoid(np.dot(X, theta)) - y)	
	return grad

	
def predict(theta, X):	
	# predict 0 if the logit value < 0.5, predict 1 if it is >= 0.5
	prediction = np.floor(sigmoid(np.dot(X, theta)) + 0.5)
	return prediction


def plotData(X,y):
	colors = ['r', 'y']
	labels = ['Type 1', 'Type 2']
	for i in range(2):
		plt.scatter(X[y==i][:,0], X[y==i][:,1], c=colors[i], marker='o', 
								s=30, label = labels[i])  
	plt.xlabel('Magnesium content')
	plt.ylabel('Flavanoid content')
	plt.title('Classification of Wines')
	plt.legend() 


def plotDecisionBoundary(theta, X, y):	
	X = X[:,[1,2]]
	plotData(X, y)

	# make predictions on a grid
	x1min, x2min = X.min(axis=0)
	x1max, x2max = X.max(axis=0)
	xx1, xx2 = np.meshgrid(np.linspace(x1min,x1max,500), np.linspace(x2min,x2max,500))
	Z = predict(theta, np.c_[np.ones(len(xx1.ravel())) ,xx1.ravel(), xx2.ravel()])
	Z = Z.reshape(xx1.shape)
	
	# plot the decision boundary according to the predictions on the grid 
	plt.contour(xx1, xx2, Z, color='b', linewidths=1.5, levels=[0])
	

def main():
	X,y = loadData('wine.data')

	
	# add intercept term to X
	m,n = X.shape	
	X = np.c_[np.ones(m), X]
	
	# set the parameters
	initial_theta = np.array(np.zeros(n+1))
	args = (X,y)
	
	# find the optimal theta using newton-conjugate-gradient method
	theta = optimize.fmin_ncg(costFunction, initial_theta, 
								fprime=costGradient, args=args)
	print 'The cost found by fmin_ncg: %f'%(costFunction(theta,*args))
	
	# visualize the decision boundary  	
	plotDecisionBoundary(theta, X, y)

	# make a prediction using the optimal theta
	prob = sigmoid(np.dot(np.array([1,12.5,25]), theta))
	print 'For a wine with magnesium content 12.5 and flavanoid content 25, we predict it'\
			' to be Type 2 with a probability of %.2f%%' %(prob*100) 
	
	# find the training accuracy
	p = predict(theta, X)
	print 'The training accuracy is %.f%%' % (np.mean(p==y) * 100)
	
	plt.show()
	
	
	
if __name__ == '__main__':
	main()
