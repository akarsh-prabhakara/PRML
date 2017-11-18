# PRML
EC447: Pattern Recognition and Machine Learning Course Project  

Vowel-only versus consonant sound classification using EEG data corresponding to speech prompts  

The data for this project was taken from Univeristy of Toronto - [KaraOne database](http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html). The objective of the main project was to identify imagined speech by finding correlations between EEG and feacial features with speech features. Instead of using multi-modal data, we use only EEG data and try to accomplish one of the sub tasks - given a processed EEG feature vector, classify whether the corresponsing speech prompt had a vowel sound or not. 

We try out different methods - Gaussian Mixture Model, Logistic Regression and Neural Networks. We implemented the GMM and Logistic regression from scratch using native Python and tested for stability too. Neural network models were built with Keras with TensorFlow backend. 

The results for the different classifiers and different parameter sweeps can be found in the project reports. The best result we obtained was for a Dense sequential network with 1 hidden layer of just 5 neurons, no regularization, batch size of 128, SGD optimizer, ReLu non linearities, and 0.01 learning rate - 61% test accuracy.  
