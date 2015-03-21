'''
We implement principal component analysis Seeds Data Set from UCI Machine Learning 
Repository. We normalize the data, project it to 2D spanned by the eigenvectors.
We also recover the projected data back to 3D. The plots of the normalized data, 
the projected data and the recovered data are available at:
https://cloud.githubusercontent.com/assets/11430119/6767374/da0fe374-d002-11e4-9b7a-ac65e00e9f8b.png
https://cloud.githubusercontent.com/assets/11430119/6767376/ddd00138-d002-11e4-9d4e-8cd92b857244.png
https://cloud.githubusercontent.com/assets/11430119/6767377/e0a4bc8c-d002-11e4-9637-80ee5a230fe6.png

The data contains measurements of geometrical  properties of kernels belonging 
to three different varieties of wheat. The data has even features and is available at 
https://archive.ics.uci.edu/ml/datasets/seeds .
We apply PCA on the data set with three features, namely: 
1) Compactness;
2) Width of Kernel;
3) Asymmetry Coefficient.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D


def loadData(filename):
	data = np.genfromtxt(filename)
	X = data[:,[2,4,5]]
	y = data[:,-1] 	
	return X,y
	
	
def featureNormalize(X):
	# normalize the features with mean=0 and std=1
	mu = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	X_norm = (X - mu) / std
	return X_norm, mu, std


def pca(X):
	# compute the covariance matrix
	sigma = np.cov(X, rowvar=0)	
	
	# compute the singular value decomposition of sigma
	U, S, Vh = linalg.svd(sigma)	
	
	return U

	
def projectData(X, U, K):
	U_reduce = U[:, range(K)]
	return np.dot(X, U_reduce)
	

def recoverData(Z, U, K):
	U_reduce = U[:, range(K)]
	return np.dot(Z, U_reduce.T)


def plot3DData(X,y,title):
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(X[:,2], X[:,1], X[:,0], c=y, cmap=plt.cm.rainbow)
	ax.azim = 180
	ax.elev = 30
	ax.set_xlabel('Compactness')
	ax.set_ylabel('Width of Kernel')
	ax.set_zlabel('Asymmetry Coefficient')
	plt.title(title)


def plot2DData(X,y,title):
	plt.figure()
	plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.rainbow)
	plt.title(title)
	
	
def main():
	X, y = loadData('seeds_dataset.txt')

	# normalize X 
	X_norm, mu, sigma = featureNormalize(X)
			
	# run PCA and get the eigenvector	
	U = pca(X_norm)
	
	# project the data onto the plane spanned by eigenvectors in 2D
	Z = projectData(X_norm, U, 2)
	
	# recover the data back to 3D
	X_rec = recoverData(Z, U, 2)
	
	# plot the normalized data, the recovered data and the projected data
	plot3DData(X_norm,y,'Original Seeds Data (Normalized)')
	plot3DData(X_rec,y, 'Seeds Data (Recovered after PCA)')
	plot2DData(Z,y, 'Seeds Data (Projected to 2D after PCA)')
	
	plt.show()
	
	
	
if __name__ == '__main__':
	main()	
