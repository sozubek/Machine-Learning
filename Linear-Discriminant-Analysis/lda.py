"""
We apply linear discriminant analysis to Waveform Data Set from UCI Machine Learning
Repository. The data contains 5000 samples each with 40 features. We compute within-class
and between-class matrices and solve the generalized eigenvalue problem. 
We pick the number of eigenvectors to be used in the projection so that 99% of the variance 
is retained. We plot the projected data. The plot is available at: 
https://cloud.githubusercontent.com/assets/11430119/7736742/07fd86c0-ff14-11e4-953b-7fa4075f4410.png
The Waveform Data Set is available at:
http://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+%28Version+2%29
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def loadData(filename):
	data = np.genfromtxt(filename, delimiter=',')
	X = data[:,:-1]
	y = data[:,-1]
	return X, y.astype(int)


def plotProjectedData(X, y):
	plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.rainbow)
	plt.title('Waveform Data After LDA')
	plt.xlabel('Linear Discriminant 1')
	plt.ylabel('Linear Discriminant 2')
	plt.show()
	
	
def main():
	# load the data from the file
	X, y = loadData('waveform-+noise.data')
	_,num_features = X.shape
	
	# compute the means
	overall_mean = np.mean(X, axis=0)
	class_means = np.array([np.mean(X[y==i], axis=0) for i in set(y)])
		
	# compute S_W: within-class scatter matrix
	class_scatters= np.array([np.dot((X[y==i] - class_means[i]).T, (X[y==i] - class_means[i])) for i in set(y)])
	S_W  = np.sum(class_scatters, axis=0)		
	
	# compute S_B: between-class scatter matrix
	coefficients = np.array([[X[y==i].shape[0]]*num_features for i in set(y)])
	mean_difference = class_means - overall_mean
	S_B = np.dot(np.multiply(coefficients, mean_difference).T, mean_difference) 
	
	# find the eigenvalues and eigenvectors of (S_W^-1)S_B
	eig_values, eig_vectors = linalg.eig(np.dot(linalg.inv(S_W), S_B)) 
	
	# sort the eigenvalues in decreasing absolute value
	# sort the corresponding eigenvectors 
	sorted_indices = np.argsort(np.absolute(eig_values))[::-1]
	sorted_eig_values =  eig_values[sorted_indices]
	sorted_eig_vectors = eig_vectors[:,sorted_indices]
		
	# pick K eigenvectors for the projection so that 99% of the variance is retained
	K = 0				
	for i in range(1, len(sorted_eig_values)+1):
		variance_retained = np.sum(sorted_eig_values[:i]) / np.sum(sorted_eig_values)
		if variance_retained >= 0.99:
			K = i
			break
	
	# project the data onto K-dimensional space
	projected_data = np.dot(X, sorted_eig_vectors[:,:K])

	# plot data after projection
	plotProjectedData(np.real(projected_data), y)
	

if __name__ == '__main__':
	main()		
