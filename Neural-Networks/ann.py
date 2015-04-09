"""
We implement an artificial neural network with regularization on the Iris data set. 
We use back propagation algorithm with batch learning to train the neural network. 
For optimization we use Newton conjugate gradient method. After training the network
we obtain 98.10% accuracy on the training set and 97.78% accuracy on the test set.
The data set is available at https://archive.ics.uci.edu/ml/datasets/Iris . 
"""

import numpy as np
from scipy import optimize
from sklearn import cross_validation


def loadData(filename):
	infile = open(filename, 'r')
	data = np.array([[item for item in line.strip().split(',')] for line in infile])		
	
	# extract the target values for the samples and the the set of target names. 
	# change the target values to integers 
	targets = data[:,-1]
	target_names = list(set(targets))
	for name in target_names:
		targets[targets == name] = target_names.index(name)
	
	targets = targets.astype(int)
	X = data[:,:-1].astype(float)

	# split the data into training and test sets with the ratio 70-30, preserving the percentage of samples for each class
	np.random.seed(0)
	cv = cross_validation.StratifiedShuffleSplit(targets, n_iter=1, test_size=0.3)

	# vectorize the target values, i.e. replace 0 by (1,0,0), 1 by (0,1,0), 2 by (0,0,1) etc.
	y = np.zeros((len(targets), len(target_names)))
	for i in range(len(targets)):
		y[i, targets[i]] = 1	
	
	for train_index, test_index in cv:
		X_train, X_test = X[train_index], X[test_index]
		y_train = y[train_index]
		targets_train, targets_test = targets[train_index], targets[test_index]

	infile.close()
	return X_train, X_test, y_train, targets_train, targets_test
	
	
def featureNormalize(X):
	# standardize the features array with mean=0 and std=1 
	mu = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	X_norm = (X - mu)/std
	return X_norm, mu, std 


def sigmoid(z):
	# sigmoid function
	return 1.0 / (1 + np.exp(-z))
	
	
def sigmoidGrad(z):
	# gradient of the sigmoid function
	return np.multiply(sigmoid(z), 1-sigmoid(z))		
	
	
def computeCostReg(params, *args):	
	# runs forward propagation and computes the cost over all samples (with regularization)

	X, y, input_layer_size, hidden_layer_size, output_layer_size, Lambda = args[0], args[1], args[2], args[3], args[4], args[5]
	m, n = X.shape 

	# Unfold the weights theta1 and theta2 from the vector "params" 
	theta1 = params[:(input_layer_size+1)*hidden_layer_size].reshape(hidden_layer_size, input_layer_size+1)
	theta2 = params[(input_layer_size+1)*hidden_layer_size:].reshape(output_layer_size, hidden_layer_size+1)	

	# when we compute the cost we do not regularize the weights coming from the bias nodes. 
	# reg_theta1 and reg_theta2 ignores these weights
	reg_theta1 = theta1[:,1:]
	reg_theta2 = theta2[:,1:] 
		
	# forward propagation
	a1 = np.c_[np.ones(m), X]		# add the bias node to layer 1
	z2 = np.dot(a1, theta1.T)		# input to layer 2
	a2 = sigmoid(z2)				# output of layer 2
	a2 = np.c_[np.ones(m), a2]		# add the bias node to layer 2
	z3 = np.dot(a2, theta2.T)		# input to layer 3
	a3 = sigmoid(z3)				# output of layer 3

	# compute the cost with regularization
	cost = (1.0/m) * np.sum(-np.multiply(y, np.log(a3)) - np.multiply(1-y, np.log(1-a3)))\
		+ Lambda / (2.0 * m) * (np.sum(np.square(reg_theta1))+ np.sum(np.square(reg_theta2)))

	return cost
	
	
def computeGradientReg(params, *args):		
	# computes the gradients of the weights using back propagation (with regularization)
	
	X, y, input_layer_size, hidden_layer_size, output_layer_size, Lambda = args[0], args[1], args[2], args[3], args[4], args[5]
	m, n = X.shape 

	# Unfold the weights theta1 and theta2 from the vector "params" 
	theta1 = params[:(input_layer_size+1)*hidden_layer_size].reshape(hidden_layer_size, input_layer_size+1)
	theta2 = params[(input_layer_size+1)*hidden_layer_size:].reshape(output_layer_size, hidden_layer_size+1)	

	# when we compute the gradient w.r.t weights, we do not regularize the weights coming from the bias nodes. 
	# reg_theta1 and reg_theta2 have these weights 0.
	reg_theta1 = np.c_[np.zeros(hidden_layer_size), theta1[:,1:]]
	reg_theta2 = np.c_[np.zeros(output_layer_size), theta2[:,1:]]
		
	# forward propagation
	a1 = np.c_[np.ones(m), X]		# add the bias node to layer 1
	z2 = np.dot(a1, theta1.T)		# input to layer 2
	a2 = sigmoid(z2)				# output of layer 2
	a2 = np.c_[np.ones(m), a2]		# add the bias node to layer 2
	z3 = np.dot(a2, theta2.T)		# input to layer 3
	a3 = sigmoid(z3)				# output of layer 3

	# back propagation
	# derivatives of the error w.r.t. inputs
	delta3 = a3 - y							# derivative of the cost function w.r.t. the input to layer 3 (i.e. z3)
	delta2 = np.multiply(np.dot(delta3, theta2)[:,1:], sigmoidGrad(z2)) 
	
	# derivatives of the error w.r.t. weights
	Delta1 = np.dot(delta2.T, a1)
	Delta2 = np.dot(delta3.T, a2)
	
	# gradients w.r.t. weights with regularization
	theta1_grad = (1.0/m) * Delta1 + (Lambda / m) * reg_theta1
	theta2_grad = (1.0/m) * Delta2 + (Lambda / m) * reg_theta2
	
	return np.r_[theta1_grad.ravel(), theta2_grad.ravel()]
	
	
def predict(theta1, theta2, X):
	m = X.shape[0]

	# find the output of the final layer
	a2 = sigmoid(np.dot(np.c_[np.ones(m), X], theta1.T))		# output of layer 2
	a3 = sigmoid(np.dot(np.c_[np.ones(m), a2], theta2.T))		# output of layer 3		
	
	# pick the class for which the corresponding output neuron gives the highest value 
	prediction = np.argmax(a3, axis=1)
	
	return prediction
	
	
def main():
	# get the training and the test data
	X_train, X_test, y_train, targets_train, targets_test = loadData('iris.data')	
		
	# normalize the features of the training and the test set
	X_train, mu, std = featureNormalize(X_train)
	X_test = (X_test - mu) / std
	
	# useful parameters
	input_layer_size = 4
	hidden_layer_size = 3
	output_layer_size = 3
	Lambda = 0.1				# regularization parameter 
	
	# randomly initialize the weights from a uniform distribution
	epsilon1 = np.sqrt(6.0/(input_layer_size+hidden_layer_size))
	epsilon2 = np.sqrt(6.0/(hidden_layer_size+output_layer_size))
	initial_theta1 = np.random.uniform(-1*epsilon1, epsilon1, size=(hidden_layer_size,input_layer_size+1))
	initial_theta2 = np.random.uniform(-1*epsilon2, epsilon2, size=(output_layer_size,hidden_layer_size+1))
	
	# set the parameters for training the neural network
	initial_params = np.r_[initial_theta1.ravel(), initial_theta2.ravel()]
	args = (X_train, y_train, input_layer_size, hidden_layer_size, output_layer_size, Lambda)

	# train the neural network
	theta = optimize.fmin_ncg(computeCostReg, initial_params, fprime=computeGradientReg, args=args)	
	
	# obtain the optimal weights 
	theta1 = theta[:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
	theta2 = theta[hidden_layer_size*(input_layer_size+1):].reshape(output_layer_size,hidden_layer_size+1)
	
	# get the predictions on the training and the test set
	prediction_train = predict(theta1, theta2, X_train)
	prediction_test = predict(theta1, theta2, X_test)
	
	# calculate the training and test set accuracy
	training_acc = np.mean(targets_train == prediction_train) * 100
	test_acc = np.mean(targets_test == prediction_test) * 100

	# report the results
	print 'The training set accuracy of the neural network: %.2f%%' % training_acc
	print 'The test set accuracy of the neural network: %.2f%%' % test_acc


if __name__ == '__main__':
	main()	
	
	
