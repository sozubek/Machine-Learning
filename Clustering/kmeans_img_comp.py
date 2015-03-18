'''
We implement K-Means algorithm for image compression. In this case the number of centroids 
become the number of colours used in the compressed image. After running the algorithm 
and computing the centroids, we replace each pixel with the centroid assigned to it.  
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def findClosestCentroids(X, centroids, m, K):	
	# The array "distances" contains the square distance of training examples to centroids 
	distances = np.zeros((m,K))
	for i in range(K):
		distances[:,i] = np.sum(np.square(X - centroids[i]), axis=1)				
	
	# assign each point to the closest centroid 	 	
	idx = np.argmin(distances, axis=1)
	
	return idx


def computeCentroids(X, idx, K):
	# return the mean of all points in the same cluster
	return np.array([np.mean(X[idx == i], axis=0) for i in range(K)])


def runkMeans(X, initial_centroids, max_iters):
	centroids = initial_centroids
	previous_centroids = centroids
	m = X.shape[0]					# number of training examples
	K = centroids.shape[0] 			# number of centroids
	
	for iter in range(max_iters):
		idx = findClosestCentroids(X, centroids, m, K)
		centroids = computeCentroids(X, idx, K)	
		if np.array_equal(centroids, previous_centroids):
			break
		else:
			previous_centroids = centroids
	
	return centroids, idx


def plotImage(A, X_recovered, K):	
	plt.subplot(121)
	plt.imshow(A)
	plt.title('Original')
	plt.xticks([])
	plt.yticks([])
	
	plt.subplot(122)
	plt.imshow(X_recovered)
	plt.title('Compressed, with %d colors.' %K)
	plt.xticks([])
	plt.yticks([])


def main():
	A = ndimage.imread('flower.png')
	img_shape = A.shape			# shape of the original image array
	
	# divide A by 255 so that all the values are in range 0-1
	A = A / 255.0

	# reshape the image into an mx3 array where m = number of pixels
	# each row will contain the Red, Green and Blue pixel values
	X = A.reshape(img_shape[0]*img_shape[1],img_shape[2])
	m = X.shape[0]			# number of pixels
	
	# set the parameters 
	K = 10					# number of centroids
	max_iters = 100			# maximum iterations for K-Means
	
	# initialize centroids randomly
	indices = np.random.random_integers(0,m-1,K)
	initial_centroids = X[indices]
	
	# run K-Means
	centroids, idx = runkMeans(X, initial_centroids, max_iters)
	
	# replace each pixel with the centroid assigned to it
	X_recovered = centroids[idx]
	X_recovered = X_recovered.reshape(img_shape[0], img_shape[1], img_shape[2])
	
	# plot the original image and the compressed image
	plotImage(A, X_recovered, K)

	plt.show()

	
if __name__ == '__main__':
	main()	
