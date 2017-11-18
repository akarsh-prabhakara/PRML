import numpy as np
import math

def logistic_regression(X_train,train_label,X_test,test_label,alpha,threshold,lam):		
	
	# Append a bias term for both train data and test data
	X_train = np.hstack((np.ones((X_train.shape[0],1)),X_train))
	X_test = np.hstack((np.ones((X_test.shape[0],1)),X_test))
	
	# Get weights by training on train data
	weights = train(X_train,train_label,alpha,threshold,lam)
	
	# Predict on test data using the weights and compute accuracy
	metrics = inference(weights,X_train,train_label,X_test,test_label)
		
	return np.concatenate([metrics,weights])
	
def train(X,label,alpha,threshold,lam):
	
	# Initialise weights
	weights  = np.random.rand(X.shape[1])
	grad = np.ones(X.shape[1])
	
	# Iterate till convergence
	while (np.sum(np.abs(grad)) >= threshold):
		#print np.sum(np.abs(grad))
		
		# Gradient computation
		grad = find_gradient(weights,X,label,lam)
		
		# Update
		weights = weights - (alpha*grad)
				
	return weights

def find_gradient(weights,X,label,lam):
	
	h = find_sigmoid(weights,X)
	grad = np.zeros(X.shape[1])
	
	# Return a gradient vector (partial derivative of cost function with respect to each feature) considering all data points 
	for l in range(X.shape[1]):
		temp = 0.0
		for i in range(X.shape[0]):
			temp = temp + ((h[i] - label[i]) * X[i,l])
		temp = temp / X.shape[0]
		if (l == 0):
			grad[l] = temp
		else:
			grad[l] = temp + (lam * grad[l])
			
	return grad
	
def find_sigmoid(weights,X):
	
	h = np.zeros(X.shape[0])
	
	# Return a vector containing output of sigmoid function with certain weights acting on each data point in X
	for i in range(X.shape[0]):
		
		# If the exponent is very large, OverflowError might occur. 
		try:
			ans = math.exp(-1.0 * X[i].dot(weights))
		except OverflowError:
			ans = float('inf')
			
		h[i] = 1.0 / (1.0 + (ans))	
		
	return h	
	
def inference(weights,X_train,train_label,X_test,test_label):

	#print "Weights =", weights 
	accuracy = np.zeros(2)
	J = np.zeros(2)
	
	# Compute train accuracy
	output = find_sigmoid(weights,X_train)
	prediction = np.zeros(X_train.shape[0])
	for i in range(X_train.shape[0]):
		if (output[i] >= 0.5):
			prediction[i] = 1
		else:
			prediction[i] = 0
	J[0] = np.sum((output - train_label)**2) / np.size(X_train,axis=0) 
	accuracy[0] = ((np.size(X_train,axis=0) - np.sum(np.absolute(prediction - train_label)))/(np.size(X_train,axis=0))*100)
	
	# Compute test accuracy
	output = find_sigmoid(weights,X_test)
	prediction = np.zeros(X_test.shape[0])
	for i in range(X_test.shape[0]):
		if (output[i] >= 0.5):
			prediction[i] = 1
		else:
			prediction[i] = 0	
	J[1] = np.sum((output - test_label)**2)	/ np.size(X_test,axis=0)
	accuracy[1] = ((np.size(X_test,axis=0) - np.sum(np.absolute(prediction - test_label)))/(np.size(X_test,axis=0))*100)
	
	metrics = np.concatenate([accuracy,J])
	return metrics