'''
We implement principal component analysis with MNIST handwritten digits dataset.
The data is available at http://www.cs.nyu.edu/~roweis/data.html . 
In particular we apply pca to the images of digit 9 which consists of 5949 training examples. 
We pick the number of principal components such that 95% variance is retained. 
We display the first 16 principal components. We also display an example of a handwritten
digit before and pca is applied. 
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy import ndimage
from scipy import linalg
from sklearn import preprocessing
from mpl_toolkits.axes_grid1 import ImageGrid


def loadData(filename):
	data = spio.loadmat(filename)
	X = data['train9'].astype(float)
	return X


def featureNormalize(X):
	# normalize the features with mean=0 and std=1
	scaler = preprocessing.StandardScaler()
	scaler.fit(X)
	X_norm = scaler.transform(X)
	return X_norm	
	
	
def pca(X):
	# compute the covariance matrix
	sigma = np.cov(X, rowvar=0)

	# compute the singular value decomposition of sigma
	U, S, Vh = linalg.svd(sigma)
	
	# return the eigenvectors and the singular values
	return U, S
	
	
def projectData(X, U, K):
	# project the data onto the eigenspace
	U_reduce = U[:,range(K)]
	return np.dot(X, U_reduce)


def recoverData(Z, U, K):
	U_reduce = U[:, range(K)]
	return np.dot(Z, U_reduce.T)
		
			
def displayData(image):
	image = image.reshape(28,28)
	plt.imshow(image, cmap = plt.cm.gray)
	

def displayExample(image1, image2):	
	plt.figure()
	
	plt.subplot(121)
	displayData(image1)
	plt.xticks([])
	plt.yticks([])
	plt.title('Original')
	
	plt.subplot(122)
	displayData(image2)
	plt.xticks([])
	plt.yticks([])
	plt.title('Recovered after PCA')	
	
	
def displayEigenvectors(U):	
	U = U.reshape(28,28,16)
	fig = plt.figure(1, (6., 6.))
	plt.title('First 16 Principal Components')
	plt.xticks([])
	plt.yticks([])
		
	grid = ImageGrid(fig, 111, nrows_ncols = (4, 4), axes_pad=0.0)
	for i in range(16):
		ax = grid[i] 
		ax.imshow(U[:,:,i], cmap = plt.cm.gray)
		ax.get_xaxis().set_visible(False)	
		ax.get_yaxis().set_visible(False)	


	
def main():
	X = loadData('mnist_all.mat')

	# normalize features with mean=0 and std=1
	X_norm = featureNormalize(X)
	
	# run PCA and get the array of eigenvectors	and the singular values
	U, S = pca(X_norm)
	
	# pick the number of principal components so that 95% of variance is retained
	K = 0				# number of principal components
	for i in range(len(S)):
		if (np.sum(S[:i+1]) / np.sum(S)) >= 0.95:
			K = i
			break	
		
	# plot the first 16 principal components 
	displayEigenvectors(U[:,:16])

	# project the images of digits to eigenspace spanned by K vectors
	Z = projectData(X_norm, U, K)
	
	# recover back the images of digits projected on the eigenspace
	X_rec = recoverData(Z, U, K)
	
	# display an example of a handwritten digit before and after PCA
	displayExample(X_norm[2], X_rec[2])


	
	plt.show()			
		
		
if __name__ == '__main__':
	main()		
