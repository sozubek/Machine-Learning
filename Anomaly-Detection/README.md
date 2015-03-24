In anomaly.py, we implement an anomaly detection algorithm with the Thyroid Data Set from UCI
Machine Learning Repository. We fit the a multivariate Gaussian distribution to 
the normal samples of the data. We select the threshold probability epsilon by 
comparing F1 scores obtained on the cross-validation set which contains both normal
and abnormal samples. The samples with probability less than epsilon are marked as 
anomalies. The plot of the normal samples with the fitted multivariate Gaussian distribution and 
the plot of the anomalies are available at:
https://cloud.githubusercontent.com/assets/11430119/6807443/e3347400-d224-11e4-811e-4b916fd163d3.png
https://cloud.githubusercontent.com/assets/11430119/6807442/e33270ba-d224-11e4-8eb8-0621a80a20ad.png

The data is available at: https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease . 
In this implementation we use the features:
1) T3-resin uptake;
2) Total serum thyroxin.  
