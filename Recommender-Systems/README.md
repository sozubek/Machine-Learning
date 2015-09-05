In recommender_system_svd.py, we use singular value decomposition to create a movie recommender system on MovieLens dataset.
- We find the best rank approximation of the ratings matrix using RMSE. 
- We interpret the latent features by exploring genre data on movies and gender data on users. 
- We calculate Precision@N metric which is the percentage of movies the user rated above a certain threshold in the recommendation list of size n. 
- To assess the ranking quality of the recommendation list we compute normalized discounted cumulative gain. This metric provides a comparison between the current recommendation list at hand and the perfectly ranked list. 
- We diversify the recommendations by using a similarity score on the list. Lower similarity scores correspond to more diverse recommendations. Here the business objective is to provide users a wider range of choices. 
- We create serendipity for the users by recommending movies that they do not expect or know about but eventually they will like. Here we use user's closest neighbours in terms of their similarity. We pick movies from the close neighbours' recommendation lists that do not appear in the current user's top recommendations.
The presentation I made on this project is available at https://drive.google.com/file/d/0B_oHTHFAMupTek5Fd0w3b1p3SU0/view?ths=true

***********************************************************
In collaborative_filtering.py, we implement a movie recommender system by creating a low rank factorization 
of the ratings matrix (also known as Funk SVD). We use regularization and solve the optimization problem 
by using L-BFGS-B method. 
The program takes movie ratings from a new user as an input and makes predictions for the unrated movies. 
Ten movies with the highest predicted ratings are displayed as top recommendations. The output of the program is 
available at https://cloud.githubusercontent.com/assets/11430119/6829884/b5a1db18-d2ec-11e4-82cf-039dee03861e.png

The dataset used is the MovieLens 100k dataset which is available at http://grouplens.org/datasets/movielens/.

************************************************************ 

In recommender_system.scala, we implement a movie recommender system using MovieLens 1M dataset using Scala and Spark.
We use the low rank factorization model provided by Spark's machine learning ibrary MLlib. We evaluate the model on the cross-validation set by RMSE and find the optimal rank for the factorization and the regularization parameter
lambda. We train the model with the best parameters and print the recommendations.

We also find movies that will create serendipity, i.e. the movies that will be pleasant surprises for the user. 
These are the movies that do not appear in user's top 10 recommendation list but the ones the user will like.
From the business perspective the aim is to avoid recommending the obvious products and reaching the long tail 
of the product catalog.
