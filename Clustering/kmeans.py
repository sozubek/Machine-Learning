'''
We implement K-Means algorithm to a real data set which is not linearly separable.
We run K-means algorithm a number of times and pick the clustering assignment with lowest 
total distance. We obtain a success rate of 89.52% with respect to the correct labels.
The data used is the Seeds Data Set from UCI. It contains measurements of geometrical 
properties of kernels belonging to three different varieties of wheat. The data has
seven features and is available at https://archive.ics.uci.edu/ml/datasets/seeds .
To provide a visualization of the clusters, we also run K-Means on the data set
with two features, namely: 
1) Area;
2) Asymmetry coefficient.  
'''

import numpy as np
import matplotlib.pyplot as plt


def loadData(filename):
	data = np.genfromtxt(filename)
	X = data[:,[0,5]]		# to run K-Means on all seven features, take X = data[:,:-1]
	y = data[:,-1]-1 	
	return X,y
	
	
def plotData(X,y):
	# plot the original data
	plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.rainbow)
	plt.title('Original Seeds Data')
	plt.xlabel('Area')
	plt.ylabel('Asymmetry Coefficient')


def plotClusters(X, centroids, idx, m, K):
	# plot the data after clustering with k-means
	plt.scatter(X[:,0], X[:,1], c=idx, cmap=plt.cm.rainbow)	
	plt.plot(centroids[:,0], centroids[:,1], 'rx', markersize=10, markeredgewidth = 2, label='Centorids')
	plt.legend()
	plt.title('Clustering with K-Means')
	plt.xlabel('Area')


def findClosestCentroids(X, centroids, m, K):	
	# The array "distances" contains the square distance of training examples to centroids 
	distances = np.zeros((m,K))
	for i in range(K):
		distances[:,i] = np.sum(np.square(X - centroids[i]), axis=1)				
	
	# assign each point to the closest centroid 	 	
	idx = np.argmin(distances, axis=1)
	
	# compute the total points-to-centroids distance 
	total_distance = np.sum(np.min(distances, axis=1))
	
	return idx, total_distance


def computeCentroids(X, idx, K):
	# return the mean of all points in the same cluster
	return np.array([np.mean(X[idx == i], axis=0) for i in range(K)])


def runkMeans(X, initial_centroids, max_iters):
	centroids = initial_centroids
	previous_centroids = centroids
	m = X.shape[0]				# number of training examples
	K = centroids.shape[0] 			# number of centroids
	distance = 0.0
	
	for iter in range(max_iters):
		idx, distance = findClosestCentroids(X, centroids, m, K)
		centroids = computeCentroids(X, idx, K)	
		if np.array_equal(centroids, previous_centroids):
			break
		else:
			previous_centroids = centroids
	
	return centroids, idx, distance


def main():
	X,y = loadData('seeds_dataset.txt')
	m = X.shape[0]			# number of training examples
	
	# set the parameters 
	K = 3				# number of centroids
	max_iters = 100			# maximum iterations for each run of K-Means
	total_runs = 200		# total number of runs for K-Means  
	
	best_result = {}
	lowest_distance = 10000.0
	
	for run in range(total_runs):
		print 'K-Means run %d/%d' %(run+1, total_runs)
		
		# randomly select an initial set of centroids from the training set
		random_indices = np.random.random_integers(0, m-1, K) 
		initial_centroids = X[random_indices]
		
		# run k-means to find the optimal centroids, the points-to-centroids assignment
		# and the total distance occurred after this assignment
		centroids, idx, distance = runkMeans(X, initial_centroids, max_iters)
		
		# store the best result w.r.t. to lowest distance occurred
		if distance < lowest_distance:
			lowest_distance = distance
			best_result['centroids'] = centroids
			best_result['idx'] = idx
	
	
	# plot original data and the result after clustering (should be omitted if we run K-Means on all seven features)
	plt.subplot(121)
	plotData(X,y)
	plt.subplot(122)
	plotClusters(X, best_result['centroids'], best_result['idx'], m, K)

	# compute success rate of k-means clustering w.r.t. the correct labels
	total_error = 0
	for i in range(K):
		occurences = np.bincount(best_result['idx'][y==i])
		error = np.sum(occurences) - occurences.max()
		total_error += error
	print 'The accuracy of K-Means is: %.2f%%' % (float(m-total_error)/m *100)
		
	plt.show()

	
	
if __name__ == '__main__':
	main()	
