In logistic_reg.py and multiclass_logistic_reg.py we analyze the Wine Data Set. 
The data contains the results of a chemical analysis of wines grown in the same region 
in Italy but derived from three different cultivars. Original data contains 13 variables.
The data is available at https://archive.ics.uci.edu/ml/datasets/Wine .

In logistic_reg.py we implement logistic regression using Newton-conjugate-gradient method. We plot the
decision boundary, calculate training accuracy (92.31%) and make predictions using the the optimal
variables. In this file we use two features of the data, namely:
1) Magnesium content;
2) Flavanoid content.
Using this information we classify two different types of wines. The plot of the data and decision boundary is available at 
https://cloud.githubusercontent.com/assets/11430119/6653486/2cbd8966-ca6b-11e4-9162-5a9fa8233ac6.png

In multiclass_logistic_reg.py we implement logistic regression to multi-class data using one-vs-all method. 
We use regularization to achieve better results. Optimization is done using conjugate-gradient method. 
We plot the decision boundary and calculate training accuracy (92.70%). 
In this file we use two features of the data, namely:
1) Malic acid;
2) Nonflavanoid phenols;
Using this information we classify three different types of wine. The plot of the data and decision boundary is
available at https://cloud.githubusercontent.com/assets/11430119/6658479/d9702684-cb40-11e4-9324-9bf307fd6923.png .

In nonlinear_logistic_reg.py we implement logistic regression to non-linear data (FLAME Data Set) using BFGS method. 
To account for the non-linearity we add polynomial features. We also use regularization to prevent
overfitting. We plot the decision boundary and calculate training accuracy (99.58%).
The plot of the data and decision boundary is available at 
https://cloud.githubusercontent.com/assets/11430119/6653492/6076201a-ca6b-11e4-9149-5ebb47d7e07b.png

The data is the FLAME data set and originates from the following article:
Fu, L. and E. Medico, FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data. 
BMC bioinformatics, 2007. 8(1): p. 3.  
The data is available at
http://cs.joensuu.fi/sipu/datasets/ .
