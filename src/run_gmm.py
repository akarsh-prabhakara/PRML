import scipy.io
import numpy as np
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse

from gmm import *

# Input
parser = argparse.ArgumentParser(description='Binary Classifier Learning Curves')
parser.add_argument('K', metavar='K', type=int, nargs=1 , help='No of times validation should be done for each percentage of training data')
parser.add_argument('C', metavar='C', type=int, nargs=1 , help='No of Gaussians per class')

args = parser.parse_args()
num_gaussians = args.C[0]

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

# Percentage to be used for training from train data (excluding validation)
percent = [0.2,1]

# Keep validation as 20% of train data
val_ratio = 0.2

# Initialize train and validation accuracy
train_accuracy = np.zeros(len(percent))
val_accuracy = np.zeros(len(percent))

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

		lda = LinearDiscriminantAnalysis(n_components=1)
		model = lda.fit(train, train_label)
		train_lda = model.transform(train)
		val_lda = model.transform(val)
		test_lda = model.transform(test)


		train = train_lda
		val = val_lda
		if (k == K-1) and (j == len(percent)-1):
				test = test_lda


		# Run classifier
		train_dict = dict()
		train_dict[1] = train[0 : len(con_train_data) ]
		train_dict[0] = train[ len(con_train_data) : ]

		val_dict = dict()
		val_dict[1] = val[0 : len(con_val_data) ]
		val_dict[0] = val[ len(con_val_data) : ]

		components = [num_gaussians,num_gaussians]
		gmm_mean, gmm_C, gmm_w, gmm_prior = gmm_train(components, train_dict)
		t_acc[k] = gmm_test(gmm_mean, gmm_C, gmm_w, gmm_prior, components, train_dict)
		v_acc[k] = gmm_test(gmm_mean, gmm_C, gmm_w, gmm_prior, components, val_dict)
		#print t_acc[k], v_acc[k]


	train_accuracy[j] = np.mean(t_acc)
	val_accuracy[j] = np.mean(v_acc)


test_dict = dict()
test_dict[1] = test[0 : len(con_test)]
test_dict[0] = test[len(con_test) : ]

test_accuracy = gmm_test(gmm_mean, gmm_C, gmm_w, gmm_prior, components, test_dict)

# Get errors from accuracies
train_error = (np.ones(len(percent))*100) - train_accuracy
val_error = (np.ones(len(percent))*100) - val_accuracy
test_error = 100.0 - test_accuracy

# Store for plotting later
# np.savez('learning_curve_gmm.npz', train_error=train_error, val_error=val_error, test_error=test_error)

print "---- OUTPUT ----"
print "Train Error =", train_error
print "Validation Error =", val_error
print "Test Error =", test_error