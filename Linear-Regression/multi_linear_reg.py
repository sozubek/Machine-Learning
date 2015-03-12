'''
We implement multivariate linear regression using gradient descent and normal equations. 
The data used is the salary data consisting of observations on six variables for 
52 tenure-track professors in a small college. Here we use three of these variables:
1) number of years in current rank
2) number of years since highest degree was earned
3) academic year salary, in thousand dollars. 
We also plot the change in cost function through the iterations of gradient descent.
Data is available at http://data.princeton.edu/wws509/datasets/#salary
'''

import numpy as np
import matplotlib.pyplot as plt



def getData(filename):
	infile = open(filename, 'r')
	data = np.array([[float(entry) for entry in line.split()]for line in infile])
	infile.close()
	X = data[:,[2,4]]
	y = data[:, -1]
	return X, y
	
	
def normalize(X):
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	X_norm = (X - mean) / std
	return X_norm, mean, std


def computeCost(X, y, theta):
	m = len(y)
	prediction = np.dot(X, theta)
	error = prediction - y
	cost = 1.0 / (2 * m) * np.dot(error, error)
	return cost
	

def gradDescent(X, y, theta, alpha, iterations):
	m = len(y)
	cost_history = np.zeros(iterations)
	for iter in range(iterations):
		theta = theta - alpha * (1.0 / m) * np.dot(X.T, (np.dot(X,theta) - y))	
		cost_history[iter] = computeCost(X, y, theta)
	return theta, cost_history
	

def plotCostHistory(cost_history, iterations):
	plt.plot(np.arange(1,iterations+1), cost_history, 'r', linewidth=2)
	plt.xlabel('Number of Iterations')
	plt.ylabel('Cost')
	plt.title('Cost History')

		
def normalEqn(X,y):		
	theta = reduce(np.dot, [np.linalg.pinv(np.dot(X.T, X)), X.T, y])
	return theta	


def main():
	# extract the data from the file 
	X, y = getData('salary.raw')	
	
	
	# normalize the features
	X_norm, mean, std = normalize(X)
	
	
	# add intercept term to X_norm
	X_norm = np.c_[np.ones(len(X_norm)), X_norm]
	
	
	# set the parameters for the gradient descent
	alpha = 0.1				# learning rate
	iterations = 100
	initial_theta = np.zeros(X_norm.shape[1])
	
	
	# find the optimal theta by gradient descent
	theta, cost_history = gradDescent(X_norm, y, initial_theta, alpha, iterations) 
	print 'Theta computed from gradient descent with alpha = %.2f and iterations = %d:' %(alpha, iterations), theta
	
	
	# plot cost vs iterations 
	plotCostHistory(cost_history, iterations)

	
	
	# predict the salary a person who earned his/her highest degree 20 years ago
	# with 10 years in current rank
	X_example = (np.array([10,20]) - mean) / std
	salary = np.dot(np.append(np.array([1]), X_example), theta)
	print 'The predicted salary of a person who earned his/her highest degree 20 years ago'\
	' with 10 years in current rank: $%.2f' %(salary)


	
	# also find the solution by using normal equations method	 
	X = np.c_[np.ones(len(X)), X]
	theta = normalEqn(X,y)
	print 'The predicted salary of the above person obtained using normal equations: %.2f' %(np.dot(np.array([1,10,20]),theta))
	
	
	plt.show()
	 
	
if __name__ == '__main__':
	main()	
	
