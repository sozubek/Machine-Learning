In svm_linear.py, we implement support vector machine with linear kernel to randomly generated data.
To find the penalty parameter C of the error term, we do stratified 10-fold cross-validation.
We find the optimal value of C according to the cross-validation error. We obtain 90% 
accuracy on the training set and a mean cross-validation score of 90.18%.   The plot of 
the data and the decision boundary is available at:
https://cloud.githubusercontent.com/assets/11430119/6743493/efa56fb2-ce70-11e4-8c32-67874d5a79d5.png

In svm_nonlinear.py, we implement support vector machine with rbf kernel to nonlinear data.  
We do stratified 15-fold cross-validation to find the best parameters C and gamma of the classifier.
The gamma parameter defines how far the influence of a single training example reaches.
The C parameter trades off misclassification of training examples against simplicity of the decision surface.
We find the optimal value of C and gamma according to the cross-validation error. We obtain 100% 
accuracy on the training set and a mean cross-validation score of 98.50%. 
The dataset is the Compound dataset which originates from the following article:
Zahn, C.T., Graph-theoretical methods for detecting and describing gestalt clusters. IEEE Transactions on Computers, 1971. 100(1): p. 68-86.  The dataset is available at http://cs.joensuu.fi/sipu/datasets/ .
The plot of the data and the decision boundary is available at:
https://cloud.githubusercontent.com/assets/11430119/6762446/c332b9c8-cf36-11e4-8928-4d0aac57dd3c.png
