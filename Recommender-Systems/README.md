In collaborative_filtering.py, we implement a movie recommender system by creating a low rank factorization 
of the ratings matrix (also known as Funk SVD). We use regularization and solve the optimization problem 
by using L-BFGS-B method. 
The program takes movie ratings from a new user as an input and makes predictions for the unrated movies. 
Ten movies with the highest predicted ratings are displayed as top recommendations. The output of the program is 
available at https://cloud.githubusercontent.com/assets/11430119/6829884/b5a1db18-d2ec-11e4-82cf-039dee03861e.png

The dataset used is the MovieLens 100k dataset which is available at http://grouplens.org/datasets/movielens/.

In recommender_system.scala, we implement a movie recommender system using MovieLens 1M dataset using Scala and Spark.
We use the low rank factorization model provided by Spark's machine learning ibrary MLlib. We evaluate the model on the cross-validation set by RMSE and find the optimal rank for the factorization and the regularization parameter
lambda. We train the model with the best parameters and print the recommendations.

We also find movies that will create serendipity, i.e. the movies that will be pleasant surprises for the user. 
These are the movies that do not appear in user's top 10 recommendation list but the ones the user will like.
From the business perspective the aim is to avoid recommending the obvious products and reaching the long tail 
of the product catalog.
