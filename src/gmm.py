import numpy as np
import math

def gmm_train(comp, X):

	prior = np.zeros(len(X.keys()))
	
	all_mean = {}
	all_C = {}
	all_w = {}
	fx = 0.0
	
	#train
	for q in X.keys():	
		fx = fx + float(np.size(X[q],axis=0)) 
			  
	for q in X.keys():
		
		x = X[q]
		prior[q] = float(np.size(X[q],axis=0))/fx
		

		K = comp[q]						# Number of clusters
		N = np.size(x,axis=0)			# Number of data points
		d = np.size(x,axis=1)			# Feature dimensions
		
		P = np.zeros(shape=(K,N))		
		w = np.array([1.0/K]*K)

		mean = {}						# Means of the Gaussians
		C = {}							# Covariance matrices of the Gaussians
		
		for k in range(0,K):
		    mean[k] = x[k]
		    C[k] = np.eye(d)
		
		#iterate     
		for l in range(0,100):
		    
		    for k in range(0,K):
		        for i in range(0,N):
		            P[k,i] = (w[k] / (math.sqrt(2 * math.pi) * math.sqrt(np.linalg.det(C[k])))) * math.exp(-0.5 * np.inner(np.inner(x[i]-mean[k],np.linalg.inv(C[k]).T),x[i]-mean[k]))
		    
		    den = np.sum(P,axis=0)   
		    
		    for k in range(0,K):
		        for i in range(0,N):
		            P[k,i] = P[k,i] / den[i]
		    
		    # Update
		    for k in range(0,K):
		        temp = np.sum(P,axis=1)
		        w[k] = temp[k] / N
		        mean[k] = np.zeros(shape=d)
		        
		        for i in range(0,N):
		            mean[k] = mean[k] + (P[k,i] * x[i])
		        
		        mean[k] = mean[k] / temp[k]
		        C[k] = np.zeros(shape=(d,d))
		        
		        for i in range(0,N):
		            C[k] = C[k] + (P[k,i] * np.outer(x[i]-mean[k],x[i]-mean[k]))
		        C[k] = C[k] / temp[k]        
		
		all_mean[q] = mean
		all_C[q] = C
		all_w[q] = w 
	
	return all_mean, all_C, all_w, prior

def gmm_test(all_mean, all_C, all_w, prior, comp, X):			
	
	labels = np.array([])
	test_data = np.array([])
	
	for q in X.keys():
		test_data = np.append(test_data,X[q])
		labels = np.append(labels,np.array([q]*len(X[q])))	

	classes = X.keys()

	#test
	output = np.zeros(np.size(test_data,axis=0))
	
	for i in range(0,np.size(test_data,axis=0)):
		out = np.zeros(len(classes))
		for j in classes:
			w = all_w[j]; mean = all_mean[j]; C = all_C[j]; 
			for k in range(0,comp[j]): 
				out[j] = out[j] + ((w[k] / (math.sqrt(2 * math.pi) * math.sqrt(np.linalg.det(C[k])))) * math.exp(-0.5 * np.inner(np.inner(test_data[i]-mean[k],np.linalg.inv(C[k]).T),test_data[i]-mean[k])))    
		
		output[i] = np.argmax(out * prior)
		          
	#accuracy
	accuracy = ((np.size(test_data,axis=0) - np.sum(np.absolute(labels - output)))/(np.size(test_data,axis=0))*100)

	return accuracy