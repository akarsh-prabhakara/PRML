import numpy as np
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras import regularizers

# Input
parser = argparse.ArgumentParser(description='Neural Networks Learning Curves')
parser.add_argument('K', metavar='K', type=int, nargs=1 , help='No of times validation test should be done for each percentage of training data')
parser.add_argument('F', metavar='F', type=int, nargs=1 , help='Feature mode. 1 indicates LDA feature only. 2 indicates features of importance only. 3 indicates LDA and features of importance. 4 indicates LDA, features of importance, and their non linearities upto 5th order. 5 All features')

args = parser.parse_args()
mode = args.F[0]

# Load data
x = np.load('../data/thinking.npz')
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
percent = [1]

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
			lda = LinearDiscriminantAnalysis()
			model = lda.fit(train, train_label)		
			train_lda = model.transform(train)
			val_lda = model.transform(val)
			test_lda = model.transform(test)
		
		# Choose features of importance and divide by their standard deviations
		# Choose features of importance for test as well to use later
		
		if mode == 2 or mode == 3 or mode == 4:
			t = 100
			train = train[:,ind[0:t]]/std[ind[0:t]]
			val = val[:,ind[0:t]]/std[ind[0:t]]
			if (k == K-1) and (j == len(percent)-1):
				test = test[:,ind[0:t]]/std[ind[0:t]]
		
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
			dim = 1
			if (k == K-1) and (j == len(percent)-1):
				test = test_lda
		elif mode == 2:
			train = train
			val = val
			dim = t
			if (k == K-1) and (j == len(percent)-1):
				test = test
		elif mode == 3:
			train = np.hstack((train,train_lda))
			val = np.hstack((val,val_lda))
			dim = t + 1
			if (k == K-1) and (j == len(percent)-1):
				test = np.hstack((test,test_lda))
		elif mode == 4:
			train = np.hstack((train,train_lda,temp_train[:,1:]))
			val = np.hstack((val,val_lda,temp_val[:,1:]))
			dim = 17
			if (k == K-1) and (j == len(percent)-1):
				test = np.hstack((test,test_lda,temp_test[:,1:]))
		elif mode == 5:
			train = train
			val = val
			test = test
			dim = 2790
		else:
			exit(1)			
			
		# Run Neural networks
		
		model = Sequential()
		model.add(Dense(5, input_dim=dim, activation='relu', kernel_regularizer=regularizers.l2(0)))
		model.add(Dense(1, activation='sigmoid'))
		
		sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
		
		model.fit(train, train_label, batch_size=128, epochs=5000, verbose=1, validation_data=(val,val_label))		  
		score = model.evaluate(train, train_label, batch_size=128)
		t_acc[k] = score[1]
		score = model.evaluate(val, val_label, batch_size=128)
		v_acc[k] = score[1]
		
	train_accuracy[j] = np.mean(t_acc)
	val_accuracy[j] = np.mean(v_acc)
	
score = model.evaluate(test, test_label, batch_size=128)	
test_accuracy = score[1]

print train_accuracy
print val_accuracy
print test_accuracy	 