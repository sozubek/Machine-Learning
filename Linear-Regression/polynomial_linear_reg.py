'''
We implement linear regression to nonlinear data using regularization. To account for the 
nonlinearity we add polynomial features to the model. We randomly split the data into training,
cross-validation and test sets with the ratio 60-20-20. We find the best values for the 
degree of the polynomial model and the regularization constant via cross-validation error. 
Finally we evaluate the test error on the test set. We also calculate the cross-validation error 
as a function of the training set size and plot the learning curve. The optimal parameters yield 
to a cross-validation error of 0.08863 and a test error of 0.009841.

The data is the Filip dataset from NIST. It available at: 
http://www.itl.nist.gov/div898/strd/lls/data/Filip.shtml  
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize



def loadData(filename):
	# load the data and randomly shuffle its rows
	data = np.genfromtxt(filename) 
	np.random.seed(30)
	np.random.shuffle(data)
	
	# split the data into training set, cross-validation set and test set with the ratio 60:20:20
	X = data[:,0] 
	y = data[:,1] 	
	m = len(y)
	X_train, X_cv, X_test = X[:np.ceil(m*0.6)], X[np.ceil(m*0.6): np.ceil(m*0.6)+np.floor(m*0.2)], X[np.ceil(m*0.6)+np.floor(m*0.2):]
	y_train, y_cv, y_test = y[:np.ceil(m*0.6)], y[np.ceil(m*0.6): np.ceil(m*0.6)+np.floor(m*0.2)], y[np.ceil(m*0.6)+np.floor(m*0.2):]
	
	return X_train, X_cv, X_test, y_train, y_cv, y_test


def plotData(X,y,color, label):
	plt.scatter(X,y, color=color, s=15, linewidth=2, label=label)
	plt.title('Polynomial Linear Regression with Regularization')			
	
	
def plotModelPoly(X, y, X_cv, y_cv, X_test, y_test, degree, optimal_theta_poly):
	# make predictions on a range using parameters of the regression model
	x = np.arange(X.min(), X.max()+0.05, 0.05)
	x_poly = polyFeatures(x,degree)
	predictions = np.dot(np.c_[np.ones(x_poly.shape[0]), x_poly], optimal_theta_poly)
	
	# plot the model using the predictions; plot the training set, the cross-validation set and the test set
	plt.figure()
	plt.plot(x, predictions, linewidth=2, color='k',linestyle='--', label='Regression Model')	
	plotData(X,y,'r', 'Training Data')
	plotData(X_cv, y_cv, 'b', 'Cross-Validation Data')
	plotData(X_test, y_test, 'g', 'Test Data')
	plt.axis('tight')
	plt.legend(loc = 'lower right')
	
	
def plotLearningCurve(error_train, error_cv):
	training_set_size = np.arange(1, len(error_train)+1)
	plt.figure()
	plt.plot(training_set_size, error_train, linewidth=2, color='r', label='Training Error')
	plt.plot(training_set_size, error_cv, linewidth=2, color='b', label= 'Cross Validation Error')
	plt.xlabel('Training Set Size')
	plt.ylabel('Error')
	plt.title('Polynomial Regression Learning Curve')
	plt.legend()
	
	
def linearRegCostFunction(theta, *args):
	X, y, Lambda = args[0], args[1], args[2]
	m, n = X.shape
	
	# we do not regularize theta_0. for that reason we use reg_theta 
	# which is the same as theta except theta_0 = 0 
	A = np.eye(n)
	A[0,0] = 0
	reg_theta = np.dot(A, theta) 
	
	# compute the cost 
	error = np.dot(X, theta) - y
	cost = 1.0 / (2 * m) * np.dot(error, error) + Lambda / (2.0 * m) * np.dot(reg_theta, reg_theta)
	return cost		

	
def linearRegGradFunction(theta, *args):
	X, y, Lambda = args[0], args[1], args[2]
	m, n = X.shape
	
	# we do not regularize theta_0. for that reason we use reg_theta 
	# which is the same as theta except theta_0 = 0 
	A = np.eye(n)
	A[0,0] = 0
	reg_theta = np.dot(A, theta) 
	
	# compute the gradient
	error = np.dot(X, theta) - y
	grad = (1.0 / m) * np.dot(X.T, error) + float(Lambda / m) * reg_theta
	return grad		


def trainLinearReg(theta, *args):
	# find the optimal parameters using bfgs method
	optimal_theta = optimize.fmin_bfgs(linearRegCostFunction, theta,
					fprime = linearRegGradFunction, args = args, maxiter = 300)
	return optimal_theta
	

def learningCurve(X_cv, y_cv, initial_theta, *args):
	X_train, y_train, Lambda = args[0], args[1], args[2]
	m = X_train.shape[0]
		
	error_train = np.zeros(m)
	error_cv = np.zeros(m)
	for i in range(m):
		# train the regression model using the first 'i' elements of the training set
		# compute the cross-validation error on the whole cross-validation set
		args_train = (X_train[:i+1], y_train[:i+1], Lambda)
		optimal_theta = trainLinearReg(initial_theta, *args_train)
		error_train[i] = linearRegCostFunction(optimal_theta, *args_train)
		
		args_cv = (X_cv, y_cv, Lambda)		
		error_cv[i] = linearRegCostFunction(optimal_theta, *args_cv)
	return error_train, error_cv


def polyFeatures(X,d):
	if d==1:
		return X
	
	X_poly = X	
	# add polynomial features to X
	for i in range(2, d+1):
		X_poly = np.c_[X_poly, X**i]
	return X_poly


def normalize(X):
	# normalize X with mean=0 and std=1
	mu = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	return (X - mu)/std, mu, std
	
	
def main():
	# get the data set and split it into three parts: 
	# training, cross-validation and testing
	X_train, X_cv, X_test, y_train, y_cv, y_test = loadData('filip.dat')

	# normalize the training set 
	# normalize the cross-validation set and the test set w.r.t. the mean and std 
	# of the training set
	X_train, mu, std = normalize(X_train)
	X_cv = (X_cv - mu) / std
	X_test = (X_test - mu) /std


	# find the best values for the degree of the polynomial model 
	# and the regularization constant Lambda using the cross-validation set
	
	# create a parameter-grid of degrees and Lambdas
	degrees = np.arange(3,20)
	Lambdas = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
	ds, ls= np.meshgrid(degrees, Lambdas)
	Z = np.c_[ds.ravel(), ls.ravel()]
	
	best_parameters = {}
	best_error_cv = 1000
	
	for i in range(Z.shape[0]):
		degree, Lambda = int(Z[i,0]), Z[i,1]
		
		# add polynomial features to the training set
		X_train_poly = polyFeatures(X_train, degree)
		
		# add intercept term to the training set 
		m,n = X_train_poly.shape 
		X_train_poly = np.c_[np.ones(m), X_train_poly]	

		# set the parameters  
		initial_theta_poly = np.zeros(n+1)
		args_poly = (X_train_poly, y_train, Lambda)
	
		# train linear regression classifier with polynomial features
		optimal_theta_poly = trainLinearReg(initial_theta_poly, *args_poly)
		
		# add polynomial features to the cross-validation set
		X_cv_poly = polyFeatures(X_cv, degree)
	
		# find the cross-validation error
		args_cv_poly = (np.c_[np.ones(X_cv_poly.shape[0]), X_cv_poly], y_cv, Lambda)		
		error_cv = linearRegCostFunction(optimal_theta_poly, *args_cv_poly)
		
		# store the best cross-validation error and the best parameters 
		if error_cv < best_error_cv:
			best_error_cv  = error_cv
			best_parameters['degree'] = degree
			best_parameters['Lambda'] = Lambda
			best_parameters['theta'] = optimal_theta_poly
			best_parameters['X_cv_poly'] = X_cv_poly
			best_parameters['initial_theta_poly'] = initial_theta_poly
			best_parameters['args_poly'] = args_poly	
	
	# plot the regression model and the training, cross-validation and test sets
	plotModelPoly(X_train, y_train, X_cv, y_cv, X_test, y_test, best_parameters['degree'], best_parameters['theta'])

	# calculate the training and cross-validation error w.r.t. the training set size
	error_train, error_cv = learningCurve(np.c_[np.ones(best_parameters['X_cv_poly'].shape[0]), best_parameters['X_cv_poly']],
										y_cv, best_parameters['initial_theta_poly'], *best_parameters['args_poly'])
	
	#plot the learning curve
	plotLearningCurve(error_train, error_cv)

	# calculate the test error
	X_test_poly =  polyFeatures(X_test, best_parameters['degree'])
	args_test_poly = (np.c_[np.ones(X_test_poly.shape[0]), X_test_poly], y_test, best_parameters['Lambda'])		
	error_test = linearRegCostFunction(best_parameters['theta'], *args_test_poly)
	
	# report the results
	print 'Best parameters are: degree = %d, lambda = %.2f' %(best_parameters['degree'], best_parameters['Lambda'])
	print 'The corresponding cross-validation error is: %f' %best_error_cv
	print 'The corresponding test error is: %f' %error_test
	
	plt.show()



if __name__ == '__main__':
	main()		
