In kmeans.py, we implement K-Means algorithm to a real data set which is not linearly separable.
We run K-means algorithm 500 times and pick the clustering assignment with lowest 
total distance (distortion). We obtain a success rate of 89.52% with respect to the correct labels.
The data used is the Seeds Data Set from UCI. It contains measurements of geometrical 
properties of kernels belonging to three different varieties of wheat. The data has
seven features and is available at https://archive.ics.uci.edu/ml/datasets/seeds .
To provide a visualization of the clusters, we also run K-Means on the data set
with two selected features, namely: 
1) Area;
2) Asymmetry coefficient.  
The plot is available at:

https://cloud.githubusercontent.com/assets/11430119/6716055/1d9d3a9c-cd79-11e4-8b31-bfee2faf89c7.png
