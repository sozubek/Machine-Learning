In kmeans.py, we implement K-Means algorithm to a real data set which is not linearly separable.
We run K-means algorithm a number of times and pick the clustering assignment with lowest 
total distance (distortion). We obtain a success rate of 89.52% with respect to the correct labels.
The data used is the Seeds Data Set from UCI Machine Learning Repository. It contains measurements 
of geometrical properties of kernels belonging to three different varieties of wheat. The data has
seven features and is available at https://archive.ics.uci.edu/ml/datasets/seeds .
To provide a visualization of the clusters, we also run K-Means on the data set
with two selected features, namely: 
1) Area;
2) Asymmetry coefficient.  
The plot is available at:
https://cloud.githubusercontent.com/assets/11430119/6716055/1d9d3a9c-cd79-11e4-8b31-bfee2faf89c7.png

In kmeans_img_comp, we implement K-Means algorithm for image compression. In this case the number of 
centroids become the number of colours used in the compressed image. After running the algorithm 
and computing the centroids, we replace each pixel with the centroid assigned to it. The original image 
and the compressed version is available at :
https://cloud.githubusercontent.com/assets/11430119/6719604/1b46e8f6-cd92-11e4-952f-d423bc428282.png
