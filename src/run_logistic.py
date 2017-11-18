import scipy.io
import numpy as np
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse

from logistic import *

# Input
parser = argparse.ArgumentParser(description='Logistic Regression Learning Curves')
parser.add_argument('K', metavar='K', type=int, nargs=1 , help='No of times validation should be done for each percentage of training data')
parser.add_argument('F', metavar='F', type=int, nargs=1 , help='Feature mode. 1 indicates LDA feature only. 2 indicates features of importance only. 3 indicates LDA and features of importance. 4 indicates LDA, features of importance, and their non linearities upto 5th order')

args = parser.parse_args()
mode = args.F[0]

# Load data
x = scipy.io.loadmat('../data/all_data.mat')
con = x['con']
vow = x['vow']

train_ratio = 0.8

# Shuffle and divide into train and test (80% and 20%, for each class)
con_shuffle = np.random.permutation(con.shape[0])
vow_shuffle = np.random.permutation(vow.shape[0])

con_train = [con[i] for i in con_shuffle[0:int(round(train_ratio*len(con)))]]
con_test = [con[i] for i in con_shuffle[int(round(train_ratio*len(con))):]]

vow_train = [vow[i] for i in vow_shuffle[0:int(round(train_ratio*len(vow)))]]
vow_test = [vow[i] for i in vow_shuffle[int(round(train_ratio*len(vow))):]]

test = np.concatenate([con_test,vow_test])
test_label = np.concatenate([[1]*len(con_test),[0]*len(vow_test)],axis=0)

# Load order of importance of features and standard deviation of corresponding features
pre = np.load('../data/feature_order.npz')
ind = pre['ind']
std = pre['std']

# Percentage to be used for training from train data (excluding validation)
percent = [0.2,1]

# Keep validation as 20% of train data
val_ratio = 0.2

# Initialize train and validation accuracy
train_accuracy = np.zeros(len(percent))
val_accuracy = np.zeros(len(percent))
train_J = np.zeros(len(percent))
val_J = np.zeros(len(percent))

# Number of times to repeat logistic regression for a certain percent of train data, for averaging
K = args.K[0]

# Get learning curves
for j in range(0,len(percent)):

	print percent[j]*100.0 ,"% of Training Data"
	t_acc = np.zeros(K)
	v_acc = np.zeros(K)
	t_J = np.zeros(K)
	v_J = np.zeros(K)
	gmm_mean = {}
	gmm_C = {}
	gmm_w = {}
	gmm_prior = []

	# Run multiple times by using a certain percent of train data for training
	for k in range(0,K):

		print "Validation Iteration ",k+1, " in progress"

		# Shuffle and divide train data into train (percent[j], for each class) and validation (20%, for each class)
		con_shuffle = np.random.permutation(len(con_train))
		vow_shuffle = np.random.permutation(len(vow_train))

		con_val_data = [con_train[i] for i in con_shuffle[0:int(round(val_ratio*len(con_train)))]]
		con_train_data = [con_train[i] for i in con_shuffle[int(round(val_ratio*len(con_train))):int(val_ratio*len(con_train) + (1-val_ratio)*len(con_train)*percent[j])]]
		
		vow_val_data = [vow_train[i] for i in vow_shuffle[0:int(round(val_ratio*len(vow_train)))]]
		vow_train_data = [vow_train[i] for i in vow_shuffle[int(round(val_ratio*len(vow_train))):int(val_ratio*len(vow_train) + (1-val_ratio)*len(vow_train)*percent[j])]]

		train = np.concatenate([con_train_data,vow_train_data],axis=0)
		val = np.concatenate([con_val_data,vow_val_data],axis=0)
		
		train_label = np.concatenate([[1]*len(con_train_data),[0]*len(vow_train_data)],axis=0)
		val_label = np.concatenate([[1]*len(con_val_data),[0]*len(vow_val_data)],axis=0)

		# Reduce the dimensions of train data to 1 using LDA, then use the same model to reduce validation data
		# Calculate reduced dimensional data for test as well to use later

		if mode == 1 or mode == 3 or mode == 4:
			lda = LinearDiscriminantAnalysis(n_components=1)
			model = lda.fit(train, train_label)
			train_lda = model.transform(train)
			val_lda = model.transform(val)
			test_lda = model.transform(test)

		# Choose features of importance and divide by their standard deviations
		# Choose features of importance for test as well to use later

		if mode == 2 or mode == 3 or mode == 4:
			train = train[:,ind[0:4]]/std[ind[0:4]]
			val = val[:,ind[0:4]]/std[ind[0:4]]
			if (k == K-1) and (j == len(percent)-1):
				test = test[:,ind[0:4]]/std[ind[0:4]]

		# Include non linear features of chosen features of importance and divide by their standard deviations
		# Find non linear features for test as well to use later

		if mode == 4:
			temp_train = np.zeros((len(train),1))
			temp_val = np.zeros((len(val),1))
			temp_test = np.zeros((len(test),1))
			for i in range(2,5):
				re_std = np.zeros(train.shape[1])
				for r in range(train.shape[1]):
					re_std[r] = np.std(train[:,r]**i)
				temp_train = np.hstack((temp_train,(train**i)/re_std))
				temp_val = np.hstack((temp_val,(val**i)/re_std))
				if (k == K-1) and (j == len(percent)-1):
					temp_test = np.hstack((temp_test,(test**i)/re_std))

		# Get final train and validation data : stack LDA feature, features of importance, non linear features of importance
		if mode == 1:
			train = train_lda
			val = val_lda
			if (k == K-1) and (j == len(percent)-1):
				test = np.hstack((np.ones((test.shape[0],1)),test_lda))

		elif mode == 2:
			train = train
			val = val
			if (k == K-1) and (j == len(percent)-1):
				test = np.hstack((np.ones((test.shape[0],1)),test))

		elif mode == 3:
			train = np.hstack((train,train_lda))
			val = np.hstack((val,val_lda))
			if (k == K-1) and (j == len(percent)-1):
				test = np.hstack((np.ones((test.shape[0],1)),test,test_lda))

		elif mode == 4:
			train = np.hstack((train,train_lda,temp_train[:,1:]))
			val = np.hstack((val,val_lda,temp_val[:,1:]))
			if (k == K-1) and (j == len(percent)-1):
				test = np.hstack((np.ones((test.shape[0],1)),test,test_lda,temp_test[:,1:]))

		else:
			exit(1)

		# Run classifier
		model = logistic_regression(train,train_label,val,val_label,0.01,5*10**-3,0)
		t_acc[k] = model[0]
		v_acc[k] = model[1]
		t_J[k] = model[2]
		v_J[k] = model[3]
		weights = model[4:]

	train_accuracy[j] = np.mean(t_acc)
	val_accuracy[j] = np.mean(v_acc)
	train_J[j] = np.mean(t_J)
	val_J[j] = np.mean(v_J)

# Use 100% training data model to test on test data
output = find_sigmoid(weights,test)
prediction = np.zeros(test.shape[0])
for i in range(test.shape[0]):
	if (output[i] >= 0.5):
		prediction[i] = 1
	else:
		prediction[i] = 0
test_J = np.sum((output - test_label)**2) / np.size(test,axis=0)
test_accuracy = ((np.size(test,axis=0) - np.sum(np.absolute(prediction - test_label)))/(np.size(test,axis=0))*100)

# Get errors from accuracies
train_error = (np.ones(len(percent))*100) - train_accuracy
val_error = (np.ones(len(percent))*100) - val_accuracy
test_error = 100.0 - test_accuracy

# Store for plotting later
# np.savez('learning_curve_log_reg.npz', train_error=train_error, val_error=val_error, train_J=train_J, val_J=val_J, test_J=test_J, test_error=test_error)


print "---- OUTPUT ----"
print "Train Error =", train_error
print "Validation Error =", val_error
print "Test Error =", test_error

print "Train J =", train_J
print "Validation J =", val_J
print "Test J =", test_J