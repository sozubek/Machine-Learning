import breeze.linalg.{DenseVector, DenseMatrix, Transpose, det, pinv}
import org.apache.spark.mllib.linalg.{Vector, Vectors, Matrix}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{SparkContext, SparkConf}


/*
We implement an anomaly detection algorithm on the Thyroid Data Set from UCI
Machine Learning Repository. We fit a multivariate Gaussian distribution to
the normal samples of the data. We select the threshold value epsilon by
comparing F1 scores obtained on the cross-validation set which contains both normal
and abnormal samples. The samples with values less than epsilon are marked as
outliers.
The data is available at: https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease .
In this implementation we use the features:
1) T3-resin uptake;
2) Total serum thyroxin.
*/


class multiVarGaussian(mu: Vector, sigma: Matrix) extends java.io.Serializable{
 // calculate parameters of the multivariate Gaussian distribution
  val n = mu.size
  val b_sigma = new DenseMatrix(rows = n, cols = n, data = sigma.toArray)
  val constant = 1.0 / math.pow(math.pow(2 * math.Pi, n.toDouble) * det(b_sigma), 0.5)
  val pinvSigma = pinv(b_sigma)

  // calculate the probability distribution function value of the vector x
  def pdf(x: Vector): Double = {
    val b_x =  new DenseVector(x.toArray)
    val b_mu = new DenseVector(mu.toArray)
    constant * math.exp(-0.5 * (Transpose(b_x - b_mu) * (pinvSigma * (b_x - b_mu))))
  }
}


object Anomaly_detection {

  def main(args: Array[String]) {

    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("Anomaly")

    val sc = new SparkContext(conf)

    // load and parse the data
    val data = sc.textFile("new-thyroid.data").map{line =>
      val fields = line.split(",")
      val label = fields.head.toInt
      val features = fields.slice(1,3).map(_.toDouble)
      (label, Vectors.dense(features))
    }

    // separate the abnormal and normal samples from each other
    val abnormalLabels = List(2, 3)
    val dataAbnormal = data.filter{ case (label, features) => abnormalLabels.contains(label)}
    val dataNormal = data.subtractByKey(dataAbnormal)

    // randomly split the normal samples into training and cross-validation sets with the ratio 80:20
    // randomly pick some abnormal samples and add them to the cross-validation set
    val splits = dataNormal.randomSplit(Array(0.8, 0.2), seed = 10L)
    val training = splits(0)
    val validation = splits(1).union(dataAbnormal.sample(withReplacement = false, fraction = 0.1, seed = 10L))

    // calculate the dataset statistics, mean and covariance, using the training set
    val trainingFeatures = new RowMatrix(training.map(_._2))
    val mu = trainingFeatures.computeColumnSummaryStatistics().mean
    val sigma =  trainingFeatures.computeCovariance()

    // calculate the probability distribution function values of the samples in the cross-validation set
    val probDist = new multiVarGaussian(mu, sigma)
    val probValidation = validation.map{ case (label, features) =>
      (label, probDist.pdf(features))
    }


    // choose a threshold value epsilon by comparing F1 scores on the cross-validation set
    // (we mark a sample x as an anomaly if pdf value of x is less than epsilon)

    // create an epsilon grid
    val maxProb = probValidation.map(_._2).max()
    val minProb = probValidation.map(_._2).min()
    val stepSize = (maxProb - minProb) / 100.0
    val epsilonGrid = minProb to maxProb by stepSize

    // calculate the F1 score corresponding to each value in the epsilon grid
    val epsilonAndF1 = epsilonGrid.map{epsilon =>
      val modelResults = probValidation.map { case (label, prob) =>
        (abnormalLabels.contains(label), prob < epsilon) match {
          case (true, true) => ("tp", 1)
          case (false, true) => ("fp", 1)
          case (true, false) => ("fn", 1)
          case (false, false) => ("tn", 1)
        }
      }
      val counts = modelResults.reduceByKey(_+_).collect().toMap.withDefaultValue(0)

      val precision = counts("tp").toDouble / (counts("tp") + counts("fp"))
      val recall = counts("tp").toDouble / (counts("tp") + counts("fn"))
      val F1 = (2 * precision * recall) / (precision + recall)

      (epsilon, F1)
    }

    // find the epsilon value providing the highest F1 score
    val bestEpsilonAndF1 = (epsilonAndF1 foldLeft (0.0, 0.0))((x, y) => if(x._2 < y._2) y else x)

    // report the best F1-score and epsilon
    val bestEpsilon = bestEpsilonAndF1._1
    val bestF1 = bestEpsilonAndF1._2
    println("Best F1-score found on the cross-validation set = " + bestF1)
    println("The threshold value epsilon that corresponds to the best F1-score = " + bestEpsilon)

    // find the samples with pdf values less than bestEpsilon in the whole dataset
    // mark them as outliers
    val outliers = data.map{ case (label, features) => (probDist.pdf(features), features)}
                       .filter{ x => x._1 < bestEpsilon }
    println("Number of outliers detected = " + outliers.count())

  }
}

