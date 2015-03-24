'''
We implement an anomaly detection algorithm with the Thyroid Data Set from UCI
Machine Learning Repository. We fit the a multivariate Gaussian distribution to 
the normal samples of the data. We select the threshold probability epsilon by 
comparing F1 scores obtained on the cross-validation set which contains both normal
and abnormal samples. The samples with probability less than epsilon are marked as 
anomalies. 
The data is available at: https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease . 
In this implementation we use the features:
1) T3-resin uptake;
2) Total serum thyroxin.  
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det, pinv
from sklearn import cross_validation 


def loadData(filename):
	data = np.genfromtxt(filename, delimiter=',')
	X = data[:,[1,2]]
	y = data[:,0]-1 	
	return X,y


def estimateGaussian(X):
	# calculate the mean of each feature and the covariance matrix
	return np.mean(X, axis=0), np.cov(X, rowvar=0)
		

def multivariateGaussian(X, mu, sigma):
	# calculate the probability of each training example with respect to the multivariate Gaussian distribution 
	m, n = X.shape
	prob = np.zeros(m)
	for i in range(m):
		prob[i] = 1.0 / ((2*np.pi)**(n/2.0) * det(sigma)**(0.5))\
		* np.exp(-0.5 * reduce(np.dot,[(X[i]-mu).T, pinv(sigma),X[i]-mu]))
	return prob

	
def selectThreshold(y_cv, prob_cv):
	# choose a threshold epsilon by comparing F1 scores on the cross-validation set.
	# (we mark a sample x as an anomaly if the probability of x is less than epsilon.)
	best_epsilon = 0
	best_F1 = 0
	stepsize = (np.max(prob_cv) - np.min(prob_cv)) / 1000.0
	for epsilon in np.arange(np.min(prob_cv), np.max(prob_cv), stepsize):
		tp = np.sum(np.logical_and(prob_cv < epsilon, y_cv == 1))
		fp = np.sum(np.logical_and(prob_cv < epsilon, y_cv == 0))
		fn = np.sum(np.logical_and(prob_cv > epsilon, y_cv == 1))
		precision = tp / float(tp + fp)
		recall = tp / float(tp + fn)
		F1 = (2.0 * precision * recall) / (precision + recall)
		if F1 >= best_F1:
			best_F1 = F1
			best_epsilon = epsilon		
	return best_epsilon, best_F1	


def plotData(X):
	plt.scatter(X[:,0], X[:,1], marker='x', color='k', s=25)

	
def visualizeFit(X, mu, sigma):
	# create a grid of values 
	x_min, y_min = np.min(X, axis=0)
	x_max, y_max = np.max(X, axis=0)
	xx, yy = np.meshgrid(np.arange(x_min,x_max,.2), np.arange(y_min,y_max,.2))
	
	# find the probabilities of the values in the grid
	Z = multivariateGaussian(np.c_[xx.ravel(), yy.ravel()], mu, sigma)
	Z = Z.reshape(xx.shape)

	# plot the data and the contour lines 
	cp = plt.contour(xx, yy, Z)
	plt.clabel(cp, inline=True, fontsize=10)
	plt.legend()
	plt.xlabel('T3-resin uptake')
	plt.ylabel('Total Serum Thyroxin')
	plt.title('Thyroid Data Set with Multivariate Gaussian (Normal Samples)')
	plotData(X)

def plotAnomalies(anomalies,X):
	plt.figure()
	plt.scatter(anomalies[:,0], anomalies[:,1], s=80, facecolor='none', edgecolors='r', label ='Anomalies') 
	plotData(X)
	plt.legend()
	plt.xlabel('T3-resin uptake')
	plt.ylabel('Total Serum Thyroxin')
	plt.title('Thyroid Data Set (All Samples)')
		
		
def main():
	# load the Thyroid data set 
	X, y = loadData('new-thyroid.data')	

	# split the samples into two groups: normal vs abnormal 
	X_normal = X[y==0]
	X_abnormal = X[y!=0]

	# make a 80-20 split of normal samples for training and cross-validation
	np.random.seed(1)
	X_normal_train, X_normal_cv = cross_validation.train_test_split(X_normal, test_size=0.2) 	
	m= X_normal_cv.shape[0]
	
	# randomly pick abnormal samples for cross-validation, add them to the cross-validation set of normal samples 
	np.random.shuffle(X_abnormal)
	X_abnormal_cv = X_abnormal[:m/5,:]
	X_cv = np.r_[X_normal_cv, X_abnormal_cv]
	y_cv = np.r_[np.zeros(m), np.ones(m/5)]
	
	# estimate the dataset statistics from normal samples
	mu, sigma = estimateGaussian(X_normal_train)
	
	# plot data with the probability contours	
	visualizeFit(X_normal_train, mu, sigma)
	
	# calculate the probabilities of samples in the cross-validation set,
	prob_cv = multivariateGaussian(X_cv, mu, sigma)
	
	# choose the threshold epsilon by comparing F1 scores on the cross-validation set
	epsilon, F1 = selectThreshold(y_cv, prob_cv)
	print 'Best F1 found using cross vailidation %f' %(F1)	
	print 'The epsilon that corresponds to the best F1 score is %e' %(epsilon)

	# find the samples with probability less than epsilon and mark them as anomalies 
	prob = multivariateGaussian(X, mu, sigma)		
	anomalies = X[prob < epsilon]
	print 'Anomalies found: %d' %(anomalies.shape[0])
	plotAnomalies(anomalies,X)
	
	plt.show()	
	
	
if __name__ == '__main__':
	main()		
