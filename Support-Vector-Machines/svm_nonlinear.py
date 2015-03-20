'''
We implement support vector machine with rbf kernel to nonlinear data.  
We do stratified 15-fold cross-validation to find the parameters C and gamma of the classifier.
The gamma parameter defines how far the influence of a single training example reaches.
The C parameter trades off misclassification of training examples against simplicity of the decision surface.
We find the optimal value of C and gamma according to the cross-validation error. We obtain 100% 
success rate on the training set and mean cross-validation score of 98.50%. 
The data set is the Compound dateset which originates from the following article:
Zahn, C.T., Graph-theoretical methods for detecting and describing gestalt clusters. 
IEEE Transactions on Computers, 1971. 100(1): p. 68-86.  
The dataset is available at http://cs.joensuu.fi/sipu/datasets/ .
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV


def loadData(filename):
	data = np.genfromtxt(filename)
	X = data[:,:-1]
	y = data[:,-1] 	
	return X,y
	
	
def plotData(X,y):
	plt.scatter(X[:,0], X[:,1], c=y, cmap = plt.cm.jet, s=25)


def plotDecisionBoundary(X, y, model, C, gamma):	
	# create a grid of values
	x1_min, x2_min = np.min(X, axis=0)
	x1_max, x2_max = np.max(X, axis=0)
	h = 400				# partition size of the grid
	xx1, xx2 = np.meshgrid(np.linspace(x1_min-1, x1_max+1, h), np.linspace(x2_min-1, x2_max+1, h))
	
	# evaluate the model on the grid
	Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
	Z = Z.reshape(xx1.shape)
	
	# plot the decision boundary according to the model's predictions on the grid
	plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired, alpha=0.5)
	plotData(X,y)
	plt.axis([-2.5,2.5,-2.5,2.5])
	plt.title('SVM with RBF Kernel (C = %.2f, Gamma = %.2f)' %(C, gamma))

		
		

def gridSearchCV(X, y, kernel):
	# set the possible values for the parameters to be examined
	values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
	parameters = dict(C=values, gamma=values)
	
	# set the K-Folds cross-validation iterator
	cv = StratifiedKFold(y=y, n_folds=15)
	
	# search for the best value of C and gamma over the possible values
	grid = GridSearchCV(SVC(kernel=kernel), param_grid=parameters, cv=cv)
	grid.fit(X, y)
	
	return grid.best_params_, grid.score(X,y)*100, grid.best_score_*100


	
def main():
	X,y = loadData('compound.txt')	 	
 	plotData(X,y)
 	
 	
 	# normalize features by removing the mean and scaling to unit variance
 	scaler = preprocessing.StandardScaler()
 	X_norm = scaler.fit_transform(X)
 	
 	# select the kernel
	kernel = 'rbf'
	
	# find the best classifier from the grid search
	params,training_score, cv_score = gridSearchCV(X_norm, y, kernel)
	C = params['C'] 
 	gamma = params['gamma']
				
 	# plot the decision boundary
 	clf = SVC(kernel=kernel, C=C, gamma=gamma)
 	clf.fit(X_norm, y)
 	plotDecisionBoundary(X_norm, y, clf, C, gamma)
 	
 	# report the success rate
 	print 'The success rate on the training set: %.2f%%' %(training_score)
 	print 'Mean cross-validation score of the model: %.2f%%' %(cv_score)
	
	plt.show()
 	
 	
 
 
if __name__ == '__main__':
	main()
