'''
We implement simple linear regression using gradient descent. The data used
is the salary data used in Weisberg(1985), consisting of observations on six variables 
for 52 tenure-track professors in a small college. Here we use two of these variables:
1) number of years since highest degree was earned
2) academic year salary, in thousand dollars. 
Apart from solving the linear regression problem, we also provide visualizations of the 
cost function in 3D and its contour plot.
Reference: S. Weisberg (1985). Applied Linear Regression, Second Edition. New York: John Wiley and Sons. Page 194.
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def getData(filename):	
	infile = open(filename, 'r')
	data = np.array([[float(entry) for entry in line.split()] for line in infile])
	infile.close
	X = data[:,-2]
	y = data[:,-1] / 1000.0
	return X,y


def plotData(X,y):
	plt.scatter(X, y, c='r', marker='x', label= 'Training data')	
	plt.xlabel('Number of Years Since Highest Degree')
	plt.ylabel('Academic Salary in $1000')


def plotRegLine(X, theta):
	plt.plot(X[:,1], np.dot(X, theta), linewidth = 2, label = 'Linear Regression')
	plt.legend(loc = 'lower right')
	plt.title('Linear Regression')

	
def computeCost(X, y, theta):
	m = len(y)
	prediction = np.dot(X, theta)
	error = prediction - y
	cost = 1.0 / (2 * m) * np.dot(error, error) 	
	return cost
	
	
def gradDescent(X, y, theta, alpha, iterations):
	m = len(y)
	for iter in range(iterations):
		theta = theta - alpha * (1.0 / m) * np.dot(X.T, (np.dot(X,theta)- y))
	return theta
	
	
def plotCost(X,y, bestTheta):
	# evaluate the cost function on a theta grid
	theta0_vals, theta1_vals = np.meshgrid(np.linspace(16,19,100), np.linspace(0.2,0.6,100))	
	thetas = np.c_[theta0_vals.ravel(), theta1_vals.ravel()]
	cost_vals = np.array([[computeCost(X, y, thetas[i])] for i in range(len(thetas))])
	cost_vals = cost_vals.reshape(theta0_vals.shape)
	
	# plot the contours of the cost function
	plt.figure()
	plt.contour(theta0_vals, theta1_vals, cost_vals, levels = np.logspace(0,2,40))
	plt.plot(bestTheta[0], bestTheta[1], 'rx', markersize=8)
	plt.xlabel('$\Theta_0$')
	plt.ylabel('$\Theta_1$')
	plt.title('Cost Function - Contours')
	
	# plot the cost function in 3D
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.plot_surface(theta0_vals, theta1_vals, cost_vals, rstride=1, cstride=1, cmap='jet')
	plt.xlabel('$\Theta_0$')
	plt.ylabel('$\Theta_1$')
	plt.title('Cost Function - Surface')
	
	
	
def main():
	# extract the data from the file and plot it
	X, y = getData('salary.raw')	
	plotData(X,y)
	
	# add a column of ones to X
	m = len(y)
	X = np.c_[np.ones(m), X]
	
	# set the parameters for the gradient descent 
	iterations = 10000
	alpha = 0.005					# learning rate
	initial_theta = np.zeros(2)	

	# run gradient descent
	theta = gradDescent(X, y, initial_theta, alpha, iterations)
	print 'Theta found by gradient descent is %f %f' %(theta[0], theta[1])
	
	# plot visualizations for the regression line and the cost function
	plotRegLine(X, theta)
	plotCost(X,y,theta)
	
	plt.show()
	

if __name__ == '__main__':
	main()	
