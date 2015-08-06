import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/*
We implement a movie recommender system using MovieLens 1M dataset which is
available at http://grouplens.org/datasets/movielens/.

We use the low rank factorization model provided by Spark's machine learning
library MLlib. We evaluate the model on the cross-validation set by RMSE
and find the optimal rank for the factorization and the regularization parameter
lambda. We train the model with the best parameters and print the recommendations.

We also find movies that will create serendipity, i.e. the movies that will
be pleasant surprises for the user. These are the movies that do not
appear in user's top 10 recommendation list but the ones the user will like.
From the business perspective the aim is to avoid recommending the obvious
products and reaching the long tail of the product catalog.
*/

object Recommender_system {

  def computeRMSE(ratings: RDD[Rating], predictFunction: RDD[(Int, Int)] => RDD[Rating]) = {
    // obtain predictions for the movies user rated
    val predictions: RDD[Rating] = predictFunction(ratings.map(x => (x.user, x.product)))

    // join the original ratings with the predictions
    val ratingsAndPredictions: RDD[(Double, Double)] = ratings.map(x => ((x.user, x.product), x.rating))
      .join(predictions.map(x => ((x.user, x.product), x.rating)))
      .values

    // calculate MSE
    val MSE: Double = ratingsAndPredictions.map{
      case (rating, prediction) => math.pow(rating - prediction, 2)
    }.mean()

    // return RMSE
    math.sqrt(MSE)
  }


  def main(args: Array[String]) {

    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("Recommender_system")

    val sc = new SparkContext(conf)

    // load user ratings
    val ratings: RDD[Rating] = sc.textFile("ratings.dat").map{line =>
      val fields = line.split("::").init
      val Array(userID, movieID) = fields.init.map(_.toInt)
      val rating = fields.last.toDouble
      Rating(userID, movieID, rating)
    }

    // load movie titles
    val movies: Map[Int, String] = sc.textFile("movies.dat").map{line =>
      val Array(movieID, movieTitle) = line.split("::").take(2)
      (movieID.toInt, movieTitle)
    }.collect().toMap

    // split the data into training and cross-validation sets
    val splits = ratings.randomSplit(Array(0.8, 0.2), seed = 0L)
    val training: RDD[Rating] = splits(0).cache()
    val validation: RDD[Rating] = splits(1).cache()


    // evaluate the model on the cross-validation set using RMSE
    // find the parameters that lead to the lowest RMSE
    val modelEvaluations =
      for(rank <- List(10, 20); lambda <- List(0.1, 1.0))
        yield {
          val model: MatrixFactorizationModel = ALS.train(training, rank, iterations = 10, lambda)
          val RMSE = computeRMSE(validation, model.predict)
          ((rank, lambda), RMSE)
        }

    println("RMSE of models with different values for rank and lambda in the format ((rank, lambda), RMSE): ")
    val sortedModelEvaluations = modelEvaluations.sortBy(_._2)
    sortedModelEvaluations.foreach(println)

    val bestRankAndLambda = sortedModelEvaluations.head._1
    val bestRank = bestRankAndLambda._1
    val bestLambda = bestRankAndLambda._2
    println("Best rank = " + bestRank)
    println("Best lambda = " + bestLambda)

    // train the model with the best parameters
    val model: MatrixFactorizationModel = ALS.train(ratings, rank = bestRank, iterations = 10, lambda = bestLambda)
    training.unpersist()
    validation.unpersist()

    //  print top 10 recommendations for a user
    val chosenUserID = 15
    val numMovies = 10
    val recommendations = model.recommendProducts(user = chosenUserID, num = numMovies)
    val recommendedMovieIDs = recommendations.map(_.product).toSet
    val recommendedMovieTitles = movies.filter{ case (movieID, movieTitle) =>
      recommendedMovieIDs.contains(movieID)
    }.values
    println("Top 10 recommendations for user 15: ")
    println("***********************************")
    recommendedMovieTitles.foreach(println)


    // pick movies to create serendipity

    // obtain "bestRank"-dimensional user-feature vectors from the model
    // sort the users with respect to the square distance between their feature
    // vectors and the the feature vector of the chosen user
    val userFeatureData = model.userFeatures
    val chosenUserFeatures = userFeatureData.filter(x => x._1 == chosenUserID).first()._2
    val userDistances = userFeatureData.map{ case (userID, features) =>
      val squaredDistance = Vectors.sqdist(Vectors.dense(features), Vectors.dense(chosenUserFeatures))
      (userID, squaredDistance)
    }.sortBy(_._2)

    // obtain a collection of movies recommended to the 20 closest neighbours
    // by choosing 10 movies for each
    val numNeighbours = 20
    val closestNeighbourIDs = userDistances.take(numNeighbours+1).tail.map(_._1)
    val recommendedNeighbourMovieIDs = closestNeighbourIDs.flatMap(userID =>
      model.recommendProducts(userID, numMovies)).map(_.product).toSet

    // get the movies in this collection that does not appear in the original recommendation list
    // print the result
    val moviesForSerendipity = recommendedNeighbourMovieIDs -- recommendedMovieIDs
    val serendipityMovieTitles = movies.filter{ case (movieID, movieTitle) =>
      moviesForSerendipity.contains(movieID)
    }.values
    println("Movies to create serendipity for user 15: ")
    println("*****************************************")
    serendipityMovieTitles.foreach(println)

  }

}
