In polynomial_linear_reg.py, we implement linear regression to nonlinear data using regularization. Optimization is achieved by BFGS method. To account for the nonlinearity we add polynomial features to the model. We randomly split the data into training, cross-validation and test sets with the ratio 60-20-20. We find the best values for the degree of the polynomial model and the regularization constant via cross-validation error. Finally we evaluate the test error on the test set. We also calculate the cross-validation error as a function of the training set size and plot the learning curve. The optimal parameters yield to a cross-validation error of 0.08863 and a test error of 0.009841. The plot of the model and the learning curve are available at:
https://cloud.githubusercontent.com/assets/11430119/6697057/4fab35da-ccc4-11e4-99f1-8b7b6ed7135d.png
https://cloud.githubusercontent.com/assets/11430119/6697059/51f4dbe8-ccc4-11e4-8612-e6dc36223077.png
The dataset used is the Filip dataset from NIST. It available at: http://www.itl.nist.gov/div898/strd/lls/data/Filip.shtml  


In linear_reg.py and multi_linear_reg.py, we implement simple linear regression and multivariate linear regression on the salary data, provided at http://data.princeton.edu/wws509/datasets/#salary .
The salary data consists of observations on six variables for 52 tenure-track professors in a small college.

In linear_reg.py, we implement simple linear regression using gradient descent. The variables used are:
1) number of years since highest degree was earned;
2) academic year salary, in thousand dollars. 
We also provide visualization of the cost function in 3D and its contour plot. The images are available at:
https://cloud.githubusercontent.com/assets/11430119/6626486/45bb9a6a-c8cd-11e4-9905-270b48a02873.png
https://cloud.githubusercontent.com/assets/11430119/6626493/51cb64ac-c8cd-11e4-8fae-eb12f4df0c66.png
https://cloud.githubusercontent.com/assets/11430119/6626491/4fb73d6c-c8cd-11e4-949c-4131de6e76c0.png

In multi_linear_reg.py, we implement multivariate linear regression using gradient descent and normal equations. 
The variables used are:
1) number of years in current rank;
2) number of years since highest degree was earned;
3) academic year salary, in thousand dollars. 
We also plot the change in cost function through the iterations of gradient descent. The image is available at:
https://cloud.githubusercontent.com/assets/11430119/6628296/a3c7d0f8-c8da-11e4-9432-99773a4a104d.png

