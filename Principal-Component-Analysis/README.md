We implement principal component analysis Seeds Data Set from UCI Machine Learning 
Repository. We normalize the data, project it to 2D spanned by the eigenvectors.
We also recover the projected data back to 3D. The plots of the normalized data, 
the projected data and the recovered data are available at: 
https://cloud.githubusercontent.com/assets/11430119/6767374/da0fe374-d002-11e4-9b7a-ac65e00e9f8b.png
https://cloud.githubusercontent.com/assets/11430119/6767376/ddd00138-d002-11e4-9d4e-8cd92b857244.png
https://cloud.githubusercontent.com/assets/11430119/6767377/e0a4bc8c-d002-11e4-9637-80ee5a230fe6.png

The data contains measurements of geometrical  properties of kernels belonging 
to three different varieties of wheat. The data has even features and is available at 
https://archive.ics.uci.edu/ml/datasets/seeds .
We apply PCA on the data set with three features, namely: 
1) Compactness;
2) Width of Kernel;
3) Asymmetry Coefficient.

In pca_mnist.py, we implement principal component analysis with MNIST handwritten digits dataset.
The data is available at http://www.cs.nyu.edu/~roweis/data.html . 
In particular we apply pca to the images of digit 9 which consists of 5949 training examples. 
We pick the number of principal components such that 95% variance is retained. 
We display the first 16 principal components. We also display an example of a handwritten
digit before and pca is applied. 
