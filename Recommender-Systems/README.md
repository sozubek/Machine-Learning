In collaborative_filtering.py, we implement a movie recommender system by creating a low rank factorization 
of the ratings matrix (also known as Funk SVD). We use regularization and solve the optimization problem 
by using L-BFGS-B method. 
The program takes movie ratings from a new user as an input and makes predictions for the unrated movies. 
Ten movies with the highest predicted ratings are displayed as top recommendations. The output of the program is 
available at https://cloud.githubusercontent.com/assets/11430119/6829884/b5a1db18-d2ec-11e4-82cf-039dee03861e.png

The dataset used is the MovieLens 100k dataset which is available at http://grouplens.org/datasets/movielens/.
It contains 100,000 movie ratings given by 943 users on 1682 movies.
