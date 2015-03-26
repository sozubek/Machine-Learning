'''
We implement a movie recommender system using collaborative filtering. We use regularization
and solve the optimization problem by using L-BFGS-B method. The program takes movie ratings
from a new user as an input and makes predictions for the unrated movies. Ten movies with the 
highest predicted ratings are displayed as top recommendations.

The dataset used is the MovieLens 100k dataset which is available at http://grouplens.org/datasets/movielens/.
It contains 100,000 movie ratings given by 943 users on 1682 movies.
'''

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def loadRatings(filename):
	infile = open(filename, 'r')
	
	# data is a 100,000 x 3 array. It contains 100,000 ratings given by 943 users 
	# on 1682 movies. The columns give the user id movie id and the rating respectively.
	data = np.array([[int(entry) for entry in line.split('\t')[:3]]for line in infile])
	ij = data[:,:2]			
	ij -=1
	values = data[:,2]
	
	# Y is the ratings array of shape num_movies x num_users.
	Y = np.zeros((1682,943), int)	
	for i in range(len(values)):
		Y[ij[i,1], ij[i,0]] = values[i]
	
	# R is the binary ratings array. R[i,j] = 0 if user j has not rated movie i and R[i,j] = 1 otherwise    
	R = Y>0
	
	infile.close()
	return Y, R
	
	
def loadMovieNames(filename):
	infile = open(filename, 'r')
	movieList = np.array([line.split('|')[1] for line in infile])	
	infile.close()
	return movieList		
	
	
def addRatings(num_movies):
	my_ratings = np.zeros(num_movies)
	my_ratings[1] = 4
	my_ratings[10] = 5
	my_ratings[21] = 4
	my_ratings[22] = 5
	my_ratings[32] = 3
	my_ratings[34] = 1	
	my_ratings[69] = 3	
	my_ratings[75] = 3	
	my_ratings[96] = 5		
	my_ratings[143] = 4
	my_ratings[153] = 2												
	return my_ratings 	
	
	
def normalizeRatings(Y, R):
	num_rows,_= Y.shape
	
	Ymean = np.array([[np.mean(Y[i,R[i,:]>0])] for i in range(num_rows)])	
	Ynorm = (Y - Ymean) * R 
	
	return Ynorm, Ymean

	
def computeCost(params, *args):
	Y, R, num_users, num_movies, num_features, Lambda\
		= args[0],args[1],args[2],args[3], args[4], args[5]

	# Unfold the arrays X and Theta from the vector params 
	X = params[:num_movies * num_features].reshape(num_movies, num_features)
	Theta = params[num_movies * num_features:].reshape(num_users, num_features)	
		
	error = (np.dot(X, Theta.T)[R>0] - Y[R>0])		
	
	# collaborative filtering cost function with regularization
	cost =	1.0/2 * np.dot(error, error) + Lambda / 2.0 *\
		(np.dot(Theta.ravel(), Theta.ravel()) + np.dot(X.ravel(), X.ravel()))
	return cost
	
	
def computeGradient(params, *args):
	Y, R, num_users, num_movies, num_features, Lambda\
		= args[0],args[1],args[2],args[3], args[4], args[5]
	
	# Unfold the arrays X and Theta from the vector params 
	X = params[:num_movies * num_features].reshape(num_movies, num_features)
	Theta = params[num_movies * num_features:].reshape(num_users, num_features)	

	error = np.multiply(np.dot(X, Theta.T), R) - Y 
	
	# collaborative filtering gradient functions with regularization
	X_grad = np.dot(error, Theta) + np.multiply(Lambda, X)
	Theta_grad = np.dot(error.T, X) + np.multiply(Lambda, Theta)		
	
	return np.r_[X_grad, Theta_grad].ravel()
	
	
def main():

	# Y is the ratings array of shape num_movies x num_users. A row of Y carries of the ratings of a specific movie 
	# by all users, a column of Y has the ratings of a specific user for all movies.
	# R is the binary ratings array. R[i,j] = 0 if user j has not rated movie i and R[i,j] = 1 otherwise     
	Y, R = loadRatings('u.data')
	
	# movieList contains the names of the movies being rated
	movieList = loadMovieNames('u.item')
	
	# add some personal ratings
	my_ratings = addRatings(Y.shape[0])

	print 
	print 'New user ratings:'
	print '******************'	
	max_len = max(len(s) for s in movieList[np.flatnonzero(my_ratings)])
	for index in np.flatnonzero(my_ratings):	
		print '%s (rated %.1f)' % (movieList[index].ljust(max_len), my_ratings[index]) 
	
	# add the new ratings to the data arrays
	Y = np.c_[my_ratings, Y]
	R = np.c_[my_ratings > 0, R]

	# normalize the data
	Ynorm, Ymean = normalizeRatings(Y,R)

	# useful values
	num_movies = Y.shape[0]
	num_users = Y.shape[1]
	num_features = 10 		
	Lambda = 10			# regularization parameter	
	
	# set initial parameters X and Theta randomly
	X = np.random.randn(num_movies, num_features)
	Theta = np.random.randn(num_users, num_features)
	
	# find the optimal solution using L-BGFS-B method
	print
	print 'The algorithm is running. Please wait...'
	initial_parameters = np.r_[X, Theta].ravel()
	args = (Y, R, num_users, num_movies, num_features, Lambda)
	result = optimize.fmin_l_bfgs_b(computeCost, initial_parameters, args = args, fprime =computeGradient)
	
	# unfold the results
	X = result[0][:num_movies*num_features].reshape(num_movies, num_features)
	Theta = result[0][num_movies*num_features:].reshape(num_users, num_features)
	
	# calculate and sort the predicted ratings for the new user
	p = np.dot(X, Theta.T)
	my_predictions = p[:,0] + Ymean.ravel()
	sorted_predictions = np.argsort(my_predictions)[::-1]
	
	print 
	print 'Top Recommendations:'
	print '*********************'
	max_len = max(len(s) for s in movieList[sorted_predictions[:10]])
	for i, index in enumerate(sorted_predictions[:10]):
		print '%2d) %s (predicted rating %.1f) ' %(i+1, movieList[index].ljust(max_len), my_predictions[index])
	

if __name__ == '__main__':
	main()
