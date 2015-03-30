''' 
We implement Naive Bayes classifier for classification using the Nursery Data Set from 
the UCI Machine Learning Repository. The data was derived from a hierarchical decision model 
originally developed to rank applications for nursery schools. It contains 8 categorical
features and 5 classes with 12690 instances. We divide the data into training and test sets with 
the ratio 80-20. We calculate conditional and prior log probabilities with Laplace smoothing. 
We make predict the class with the maximum posterior probability. We obtain a 88.46% accuracy
on the test set.
The data set is available at http://archive.ics.uci.edu/ml/datasets/Nursery.
'''

import numpy as np
from sklearn import cross_validation


def loadData(filename):
	infile = open(filename, 'r')
	data = np.array([[item for item in line.strip().split(',')] for line in infile])		
	return data[:,-1], data[:,:-1]
	
	
def calculateCondProbs(data, feature_names):
	num_samples, num_features = data.shape
	
	# calculate the conditional probability of each feature given the class
	cond_probs = {} 
	for i in range(num_features):	
		for name in feature_names[i]:
			num_instances = (data[data[:,i] == name].shape[0])
			# calculate log probabilities with Laplace smoothing
			cond_probs[name] = np.log(float(num_instances + 1) / (num_samples + len(feature_names[i])))
	return cond_probs
		
		
def predict(data, prior_probs, conditional_probs, class_names):
	num_samples, num_features = data.shape 
	num_classes = len(class_names)
	posterior_probs = np.zeros((num_samples,num_classes))
	
	# calculate the posterior probability of each test sample for each class
	for i in range(num_samples):
		for class_name in class_names:
			j = class_names.index(class_name)
			for feature_name in data[i]:
				posterior_probs[i,j] += conditional_probs[class_name][feature_name] 
		posterior_probs[i,j] += prior_probs[class_name] 	
	
	# predict the class with the highest posterior probability
	predictions = np.argmax(posterior_probs, axis=1)
	return predictions
	
		
def main():
	classes, data = loadData('nursery.data')
	_,num_features = data.shape	
	
	# get the names of the classes and the features 
	class_names = list(set(classes))	
	feature_names = []
	for i in range(num_features):
		feature_names.append(set(data[:,i]))

	# split the data into training and test sets with the ratio 80-20
	np.random.seed(6)
	data_train, data_test, classes_train, classes_test = cross_validation.train_test_split(data,classes,test_size=0.2)
	num_samples,_ = data_train.shape	

	# calculate the prior and conditional log probabilities for each class with Laplace smoothing
	prior_probs = {}
	conditional_probs = {}
	for name in class_names:
		class_data = data_train[classes_train == name]
		prior_probs[name] =  np.log(float(class_data.shape[0]+1)/(num_samples + len(class_names)))
		conditional_probs[name] = calculateCondProbs(class_data, feature_names)

	# make predictions on the test set, calculate the accuracy 
	predictions = predict(data_test, prior_probs, conditional_probs, class_names)
	for class_name in class_names: 
		classes_test[classes_test==class_name] = class_names.index(class_name)
	accuracy = np.mean(predictions == classes_test.astype(int))*100
	
	print 'The accuracy of Naive Bayes on the test set: %.2f%%' %(accuracy)


if __name__ == '__main__':
	main()		
