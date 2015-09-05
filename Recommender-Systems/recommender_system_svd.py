import numpy as np
from scipy import sparse
import scipy.linalg as linalg
import operator
import matplotlib.pyplot as plt



def loadRatings(filename):
	infile = open(filename, 'r')

	# randomly pick indices the training, cross-validation and test sets. 	
	# the ratio is 60-20-20
	seed_no = 9
	np.random.seed(seed_no)
	cv_test_indices = np.random.choice(range(100000), size=40000, replace=False)
	train_indices = np.setdiff1d(np.arange(100000), cv_test_indices)
	cv_indices = cv_test_indices[:20000]
	test_indices = cv_test_indices[20000:]

 	# put the corresponding indices and values in arrays
	data = np.loadtxt(filename, dtype=str)[:,:3]		
	data = data.astype(float)
	ij = data[:,:2]			
	ij -=1
	values = data[:,2]

	ij_train = ij[train_indices]		
	values_train = values[train_indices]

	ij_cv = ij[cv_indices]		
	values_cv = values[cv_indices]

	ij_test = ij[test_indices]			
	values_test = values[test_indices]
		
	# construct the ratings arrays for training, cross-validation and test data
	R_train = np.zeros((943,1682))	
	for i in range(len(values_train)):
		R_train[ij_train[i,0], ij_train[i,1]] = values_train[i]

	R_cv = np.zeros((943,1682))	
	for i in range(len(values_cv)):
		R_cv[ij_cv[i,0], ij_cv[i,1]] = values_cv[i]

	R_test = np.zeros((943,1682))	
	for i in range(len(values_test)):
		R_test[ij_test[i,0], ij_test[i,1]] = values_test[i]
	
	
	# construct binary rating arrays 
	B_train = R_train > 0
	B_cv = R_cv > 0
	B_test = R_test > 0
	
	return R_train, R_cv, R_test, B_train, B_cv, B_test
	
	
def loadMovieInfo(filename):
	'''
	Loads the movie names and the genre information	
	'''
	infile = open(filename, 'r')
	movieNames = np.array([line.split('|')[1] for line in infile])	
	infile.close()
	infile2 = open(filename, 'r')
	movieGenres = np.asarray([line.strip().split('|')[-19:] for line in infile2], int)	
	infile2.close()
	return movieNames, movieGenres		
	
	
def loadGenreNames(filename):
	infile = open(filename, 'r')
	genreNames = np.array([line.split('|')[0] for line in infile])	
	return genreNames


def loadUserInfo(filename):	
	'''
	Loads user's genders
	'''
	infile = open(filename, 'r')
	genders = np.array([line.split('|')[2] for line in infile])	
	genders = (genders == 'F')
	return genders
	

def getFeatureCompositions_Movies(feature_index, V, movieGenres, movieNames):
	'''
	Given a feature obtains the genres of the top 15 movies with the largest coefficients
	in that feature. Uses TF-IDF to assess the relevance of these genres.
	'''
	genreNames = loadGenreNames('u.genre')
	idfGenres = np.log(np.sum(movieGenres) / np.sum(movieGenres, axis=0)) 

	# sort the movies wrt the their scores on the movie-feature matrix
	sorted_movies = np.argsort(V[:,feature_index])[::-1]

	# get the top n movies with the largest coefficients
	n = 15
	top_n_indices = sorted_movies[:n]
	
	# for these movies get the genre vectors and apply tfidf
	top_n_genres = movieGenres[top_n_indices]
	genre_freq = np.sum(top_n_genres, axis=0)
	tfidf = genre_freq * idfGenres
	
	# get the most relevant genres in this vector
	top_n_tfidf = np.argsort(tfidf)[::-1]
	
	
	# report the result	
	max_len = max(len(s) for s in genreNames[top_n_tfidf[:3]])
	print
	print 'The genre composition for feature %d among top %d movies:' %(feature_index + 1, n)
	print '**********************************************************'
	for i, index in enumerate(top_n_tfidf[:3]):
		print '%s (%.2f%%) (%d votes)' %(genreNames[index].ljust(max_len),  tfidf[index]/np.sum(tfidf) * 100, genre_freq[index])
	print 
	print 'Movies with the highest scores for feature 7'
	print '*********************************************'
	print movieNames[top_n_indices]
	print


def getFeatureCompositions_Genders(feature_index, U):
	'''
	Given a feature obtains the genders of the top 10 people with the largest coefficients
	in that feature. Uses TF-IDF to assess the relevance of the gender information.
	'''
	
	userGenders = loadUserInfo('u.user')	 	
	
	num_total = float(len(userGenders))
	num_females = np.sum(userGenders)
	num_males = len(userGenders) - np.sum(userGenders)
	female_idf = num_total / num_females
	male_idf = num_total / num_males	
	
	
	# sort the users wrt the their scores on the user-feature matrix
	user_features = np.argsort(U, axis=0)[::-1]
	
	# get the top n users with the largest coefficients
	n = 10
	best_feature_indices = user_features[:,feature_index][:n]

	female_freq = np.sum(userGenders[best_feature_indices])
	male_freq = n - np.sum(userGenders[best_feature_indices])
	
	female_tfidf = female_freq * female_idf
	male_tfidf = male_freq * male_idf
	
	# report the result
	print 'The gender composition of feature %d among top %d people:' %(feature_index + 1, n)
	print '*********************************************************'
	print 'Female (%.2f%%) (%d females)' %(female_tfidf / (female_tfidf + male_tfidf) * 100, female_freq)
	print 'Male   (%.2f%%) (%d males)' %(male_tfidf / (female_tfidf + male_tfidf) * 100, male_freq)
	print 
	
	
def normalizeRatings(R, B):
	# mean normalize user ratings
	Rmean_user = (np.sum(R, axis=1) / np.sum(B, axis=1))[:,np.newaxis]	
	std = np.std(R, axis=1)[:,np.newaxis]
	Rnorm_user = (R - Rmean_user) * B
	
	R_norm_user_std = ((R - Rmean_user) * B) / std
	
	
	# mean normalize movie ratings
	num_movies = R.shape[1]
	Rmean_movie = np.zeros(num_movies)
	for movie_id in range(num_movies):
		vote_count = np.count_nonzero(B[:,movie_id])
		if vote_count==0:
			continue
		Rmean_movie[movie_id] = (np.sum(R[:,movie_id]) / vote_count)					
	
	Rnorm_movie = (R - Rmean_movie) * B
	
	return Rnorm_user, Rmean_user, R_norm_user_std, std, Rnorm_movie, Rmean_movie
			

def svd(R):
	'''
	Returns singular value decomposition of the ratings matrix
	'''
	U, S, Vt = linalg.svd(R, full_matrices=False)
	k = len(S) 
	S = linalg.diagsvd(S, k, k)
	return U, S, Vt


def reduceRank(U, S, Vt, n):
	'''
	Reduces the rank of the matrices in SVD factorization to the specified number n.
	'''
	# reduce the matrices to rank n
	U_reduce = U[:, range(n)]
	Vt_reduce = Vt[range(n), :]
	S_reduce = S[range(n),:][:,range(n)] 

	return U_reduce, S_reduce, Vt_reduce


def computeRMSE(R, B, predictions):	
	'''
	Computes the root mean square error between the actual ratings and the predictions 
	'''
	
	# ************ This is MSE where each user counts the same		
	sum_errors = 0
	voter_count = 0
	squared_error_matrix = np.square(R - np.multiply(predictions,B))
	# calculate the mean squared error for each user
	for user_id in range(R.shape[0]):	
		num_user_votes = np.count_nonzero(B[user_id])
		if num_user_votes == 0:
			continue
		mean_user_error = np.sum(squared_error_matrix[user_id]) / num_user_votes
		voter_count += 1
		sum_errors += mean_user_error
		
	# calculate the mean error among users
	mean_error = sum_errors / voter_count
	
	'''	
	# ************ This is MSE where each vote counts the same		
 	num_nonzero = np.count_nonzero(B)
	mean_error = np.sqrt(np.sum(np.square(R - np.multiply(predictions,B))) / num_nonzero)
	'''
	
	return mean_error
	
						
def get_best_RMSE_model_SVD(R_cv, B_cv, Rmean_movie, U, S, Vt, ranks):
	'''
	Finds the rank of the SVD approximation which produces the least root mean square error
	'''
	best_RMSE = np.inf
	best_rank = -1
	RMSE_list = []
	for n in ranks:
		# reduce the matrices to rank n
		U_reduce, S_reduce, Vt_reduce = reduceRank(U, S, Vt, n)

		# get the predictions
		predictions = reduce(np.dot, [U_reduce, S_reduce, Vt_reduce]) + Rmean_movie
		
		# calculate the mean squared error between actual ratings and predictions
		error = computeRMSE(R_cv, B_cv, predictions)	
		RMSE_list.append(error)
			
		# check if the error is best so far
		if error < best_RMSE:
			best_RMSE = error
			best_rank = n	
			
	return 	best_rank, best_RMSE, RMSE_list		
	
	
	
def computePrecision(R_cv, B_cv, predictions, threshold, n):			
	'''
	Computes Precision@N metric on the cross-validation set.
	Precision@N is the percentage of movies the user rated above threshold in the recommendation list of size n
	'''
	
	cv_predictions = np.multiply(predictions, B_cv)
	sorted_predictions = np.fliplr(np.sort(cv_predictions))[:,:n]	
	top_indices = np.fliplr(np.argsort(cv_predictions))[:,:n]
	
	num_users = R_cv.shape[0]
	precision = np.zeros(num_users, dtype=float)
	
	for user_id in range(num_users):
		user_liked = np.ceil(threshold)<= np.trim_zeros(R_cv[user_id,top_indices[user_id]]) 
		user_disliked = np.trim_zeros(R_cv[user_id,top_indices[user_id]]) <= np.floor(threshold)
		
		# we think that a recommendation is good if the predicted rating 
		# is grater than threshold
		not_recommended = np.trim_zeros(sorted_predictions[user_id]) < threshold
		recommended = np.trim_zeros(sorted_predictions[user_id]) >= threshold
		
		tp = np.count_nonzero(np.logical_or(not_recommended, user_liked))
		fp = np.count_nonzero(np.logical_and(recommended, user_disliked))
		precision[user_id] = float(tp) / (tp+fp)
	
	mean_precision = np.mean(precision)
	return mean_precision


def get_best_precision_model_SVD(R_cv, B_cv, U, S, Vt, Rmean_movie, ranks):
	'''
	Finds the rank of the SVD approximation which produces the best precision@n score
	'''
	list_length = 10	# length of the recommendation list
	threshold = 3.5		# take predictions above threshold as a recommendation for good movies
	
	best_precision = 0
	best_rank = -1
	precision_list = []
	for n in ranks:
		# reduce the matrices to rank n
		U_reduce, S_reduce, Vt_reduce = reduceRank(U, S, Vt, n)

		# get the predictions
		predictions = reduce(np.dot,[U_reduce, S_reduce, Vt_reduce]) + Rmean_movie	
		
		# calculate the precision
		precision = computePrecision(R_cv, B_cv, predictions, threshold, list_length)		
		precision_list.append(precision)
		
		# check if the precision is the best so far
		if precision > best_precision:
			best_precision = precision
			best_rank = n		
				
	return 	best_rank, best_precision, precision_list		



def computeNDCG(R_cv, B_cv, predictions):
	''' 
	Calculates the normalized discounted cumulative gain for top n movies in the
	recommendation list.
	'''
	
	n = 10 	# the number of movies in our prediction list to compute NDCG on
	cv_predictions = np.multiply(predictions, B_cv)
	top_indices = np.fliplr(np.argsort(cv_predictions))[:,:n]
	
	num_users = R_cv.shape[0]
	discount_factors = 1 / np.maximum(1, np.log2(np.arange(1,n+1)))	
	NDCG = np.zeros(num_users, dtype=float)	
	
	for user_id in range(num_users):
		actual_ratings = R_cv[user_id,top_indices[user_id]]
		sorted_actual_ratings = np.sort(actual_ratings)[::-1]
		DCG = np.dot(actual_ratings, discount_factors)
		IDCG = np.dot(sorted_actual_ratings, discount_factors)
		NDCG[user_id] = DCG / IDCG
							
	mean_NDCG = np.mean(NDCG)
	return mean_NDCG



def diversifyList(user_id, predictions, U, V):
	'''
	Diversifies the recommendation list by using a intra-list similarity metric.
	'''
	
	n = 20	# no of movies to be used in diversification
	k = 10	# no of movies to be in the output list

	# sort the top n predictions
	sorted_pred = np.argsort(predictions[user_id])[::-1]
	rec_list = sorted_pred[:n]		# original recommendations in decreasing ratings order

	# get correlations of movies from their feature vectors	
	movie_similarities = np.corrcoef(V)

	# the new recommendation list after diversification
	new_list = []
	
	# put the first movie of the original list to the new list
	new_list.append(rec_list[0])
	rec_list = np.delete(rec_list,0)

	# pick the rest of the movies that will lead to the smallest intralist similarity
	# at each step
	intralist_sim = 0	
	for i in range(k-1):
		list_sim_with_movie = []
		for movie in rec_list:
			similarity = 0
			for item in new_list:
				similarity += movie_similarities[movie, item]
			list_sim_with_movie.append(similarity)
				
		index_min = np.argmin(list_sim_with_movie)
		new_list.append(rec_list[index_min])
		rec_list = np.delete(rec_list, index_min)
		min_sim = np.min(list_sim_with_movie)
		intralist_sim += min_sim
		
	intralist_sim = intralist_sim / (k*(k-1)/2.0)	
	
	return new_list	



def createSerendipity(predictions, user_corr, R_train_mean_movie, movieNames):
	'''
	Creates serendipity by choosing movies from a user's closest neighbours.
	Similarity of users are computed using Pearson correlation. Finds the movies which
	are in the user's neighbour's top 20 list but not in his/her top 20 list.
	'''
	n = 20		# pick a movie outside the user's top n
	m = 20		# consider top m movies of the neighbours for serendipity 
	user_id = 10
	recommended_movies = (np.argsort(predictions[user_id])[::-1])[:n]	
	
	# pick top 10 similar users to the given user
	similar_users = (np.argsort(user_corr[user_id])[::-1])[1:11]

	# search the top m recommendations of the neighbours to find movies
	# that do not appear in the top n recommendations of the given user
	additional_movies = {}
	for neighbour in similar_users:
		neighbour_movies = (np.argsort(predictions[neighbour])[::-1])[:m]	
		difference = np.setdiff1d(neighbour_movies, recommended_movies)
		for movie in difference:
			additional_movies[movie] = max(predictions[user_id, movie] - R_train_mean_movie[movie], 0) * (predictions[user_id, difference[0]] >= 3.5)
	
	# print 'THE ORIGINAL LIST:'
	printList(user_id, movieNames, predictions, recommended_movies, 10)
	print 
		
	sorted_additional = sorted(additional_movies.items(), key=operator.itemgetter(1))
	max_len = max(len(s) for s in [movieNames[sorted_additional[-1][0]], movieNames[sorted_additional[-2][0]], movieNames[sorted_additional[-3][0]] ])
	print 
	print 'Movies to create serendipity:'
	print '******************************'	
	print '%s	predicted rating:%.2f  		average movie rating:%.2f' %(movieNames[sorted_additional[-1][0]].ljust(max_len), predictions[user_id, sorted_additional[-1][0]], R_train_mean_movie[sorted_additional[-1][0]]) 
	print '%s	predicted rating:%.2f 		average movie rating:%.2f' %(movieNames[sorted_additional[-2][0]].ljust(max_len), predictions[user_id, sorted_additional[-2][0]], R_train_mean_movie[sorted_additional[-2][0]]) 
	print '%s	predicted rating:%.2f 		average movie rating:%.2f' %(movieNames[sorted_additional[-3][0]].ljust(max_len), predictions[user_id, sorted_additional[-3][0]], R_train_mean_movie[sorted_additional[-3][0]]) 
	
	


def printList(user_id, movieNames, predictions, sorted_indices, length):
	pred = predictions[user_id]
	print 'Top Recommendations:'
	print '*********************'
	max_len = max(len(s) for s in movieNames[sorted_indices[:length]])
	for i, index in enumerate(sorted_indices[:length]):
		print '%2d) %s (predicted rating %.1f) ' %(i+1, movieNames[index].ljust(max_len), pred[index])



def printDiversifiedList(user_id, movieNames, predictions, recommended_movies, new_rec_list, length):
	print 'THE ORIGINAL LIST:'
	printList(user_id, movieNames, predictions, recommended_movies, length)
	print 
	print '*****************************************************************'
	print 
	print 'THE LIST AFTER DIVERSIFICATION:'
	printList(user_id, movieNames, predictions, new_rec_list, length)
	

def plotRMSE(errors):
	plt.plot(errors[:,0], errors[:,1], linewidth=3, color='r')
	plt.xlabel('Ranks')
	plt.ylabel('RMSE')
	plt.title('RMSE on Cross-Validation Set')
	plt.show()	

def plotPrecision(precisions):
	plt.plot(precisions[:,0], precisions[:,1], linewidth=3, color='r')
	plt.xlabel('Ranks')
	plt.ylabel('Precision')
	plt.title('Precision@10 on Cross-Validation Set')
	plt.show()		
	
def plotRMSE_USER(errors):
	plt.plot(errors[:,0], errors[:,1], linewidth=3, color='r')
	plt.xlabel('Neighborhood Size')
	plt.ylabel('RMSE')
	plt.title('RMSE on Cross-Validation Set')
	plt.show()	

	
def main():
	# load ratings from the training cross-validation and test sets 
	R_train, R_cv, R_test, B_train, B_cv, B_test = loadRatings('u.data')	
		
	# load movie info 
	movieNames, movieGenres = loadMovieInfo('u.item')

	# mean normalize the training data wrt user ratings 
	R_train_norm_user, R_train_mean_user, R_train_norm_user_std, user_std, R_train_norm_movie, R_train_mean_movie = normalizeRatings(R_train, B_train)
		
	# apply singular value decomposition to training set
	U, S, Vt = svd(R_train_norm_movie)
	

	'''FIND THE SVD MODEL WITH THE BEST RANK WRT RMSE'''
	'''
	# find the model with the best RMSE by cross-validation, normalizing with mean user ratings
	# The lowest RMSE comes with rank 14 approximation with error 1.072588 

	ranks = np.arange(5,31)
	best_rank, best_RMSE, RMSE_list = get_best_RMSE_model_SVD(R_test, B_test, R_train_mean_movie, U, S, Vt, ranks)
	print 'The lowest RMSE comes with rank %d approximation with error %f' %(best_rank, best_RMSE)
	plotRMSE(np.c_[ranks, RMSE_list])
	'''
		
	'''PARAMETERS IN THE OPTIMAL MODEL'''
	
	best_rank = 14
	U, S, Vt = reduceRank(U, S, Vt, best_rank)
	predictions_SVD = reduce(np.dot, [U, S, Vt]) + R_train_mean_movie
	

			
	''' INTERPRET SOME OF THE FEATURES'''
	'''
	# get an interpretation for the features
	getFeatureCompositions_Movies(6, Vt.T, movieGenres, movieNames)	
	getFeatureCompositions_Genders(6, U) 
	'''
	

	''' FIND THE SVD MODEL WITH THE BEST RANK WRT PRECISION'''
	'''
	# find the model with the best precision on the cross-validation set
	ranks = np.arange(5,26)
	best_rank, best_precision, precision_list = get_best_precision_model_SVD(R_test, B_test, U, S, Vt, R_train_mean_movie, ranks)
	print 'The best mean precision comes with rank %d approximation with precision %.2f%%' %(best_rank, best_precision*100)
	plotPrecision(np.c_[ranks, precision_list])
	'''
	
	
	''' COMPUTE NORMALIZED DISCOUNTED CUMULATIVE GAIN (NDCG)'''
	'''
	# compute the NDCG on the test set
	predictions = predictions_SVD
	meanNDCG = computeNDCG(R_test, B_test, predictions)
	print 'The normalized discounted cumulative gain of the model: %.3f' %meanNDCG
	'''


	'''	DIVERSIFY THE RECOMMENDATION LIST'''
	'''
	predictions = predictions_SVD
	user_id = (np.argsort(U[:,6]))[::-1][0]		
	new_rec_list = diversifyList(user_id, predictions, U, Vt.T)
	recommended_movies = np.argsort(predictions[user_id])[::-1]
	printDiversifiedList(user_id, movieNames, predictions, recommended_movies, new_rec_list, 10)	
	'''
	
	
	''' PICK MOVIES FOR SERENDIPITY'''
	'''
	user_corr = np.corrcoef(U)		 	 # calculate the correlations between users
	createSerendipity(predictions_SVD, user_corr, R_train_mean_movie, movieNames)
	'''

	
if __name__ == '__main__':
	main()	
	
	

	
