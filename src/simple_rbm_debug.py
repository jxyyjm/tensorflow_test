#!/usr/bin/python
# -*- coding:utf-8 -*-

# @time      : 2017-05--4
# @author    : yujianmin
# @reference : An Introduction to Restricted Boltzmann Machine
# @what to-do: try to build a RBM model ## RBM could get a distribution from training data ##


from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn import datasets
from sklearn import linear_model
try: from sklearn.cross_validation import train_test_split
except: from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

# RBM method #
'''
	n-nodes   * * *    hidden-layer, c bias
	         /|\|/|\ 
	m-nodes * * * * *  input-layer , b bias

	1. init parameter, W_{n,m}=0, b_{m,1}=0, c_{n,1}=0
	2. v^(0) = input(x)
	   2.1 gibbs sample
	       for i=1,2,...,n, do sample h_i^(0)~p(h_i|v^(0)) # sample h given v
		   for j=1,2,...,m, do sample v_j^(1)~p(v_j|h^(0)) # sampel v given h
	   2.2 update parameter
	       W_{i,j} = W_{i,j} + p(hi=1|v^(0))*vj^(0) - p(hi=1|v^(1))*vj^(0)
		   b_{j,1} = b_{j,1} + vj^(0) - vj^(1)
		   c_{i,1} = c_{i,1} + p(hi=1|v^(0)) - p(hi=1|v^(1))
'''
mnist   = input_data.read_data_sets('MNIST_data/', one_hot=True)

trainX  = mnist.train.images
one_pos = np.where(trainX > 0)
trainX[one_pos] = 1
trainY  = mnist.train.labels
trainY1 = np.argmax(trainY, 1)

testX   = mnist.test.images
one_pos = np.where(testX > 0)
testX[one_pos] = 1
testY   = mnist.test.labels
testY1  = np.argmax(testY, 1)

def gibbs_sample(prob):
	#print ('in gibbs_sampe', prob.shape)
	gibbs_sample= np.random.binomial(1, prob)
	return gibbs_sample
	# notice :: here will give an error, like this 'setting an array element with a sequence'
	#           because [None, n_feaure], tensor-variable will lead to fail 
def sigmoid(x):
	return 1/(1+np.exp(-x))

def CD_Grad_one_step(x, W, vb, hb):
	h_prob_0 = sigmoid(np.dot(x, W) + hb) 
	# [None, n_feature] * [n_feature * n_hidden] = [None, n_hidden]
	h_samp_0 = gibbs_sample(h_prob_0)            # [None, n_hidden] 
	v_prob_1 = sigmoid(np.dot(h_samp_0, np.transpose(W)) + vb)
	# [None, n_hidden] *[n_feature * n_hidden]^T = [None, n_feature]
	v_samp_1 = gibbs_sample(v_prob_1)            # [None, n_feature]
	h_prob_1 = sigmoid(np.dot(v_samp_1, W) + hb)

	grad_W = np.dot(np.transpose(x), h_prob_0)/x.shape[0] \
		   - np.dot(np.transpose(v_samp_1),h_prob_1)/x.shape[0]
	   # [None, n_feature] * [None, n_hidden]^T  = [n_feature, n_hidden]
	grad_vb= np.mean(x - v_samp_1, 0)
	grad_hb= np.mean(h_prob_0 - h_prob_1, 0)

	return grad_W, grad_vb, grad_hb

def error(x, W, vb, hb):
	h_prob_0 = sigmoid(np.dot(x, W) + hb) 
	h_samp_0 = gibbs_sample(h_prob_0) 
	v_prob_1 = sigmoid(np.dot(h_samp_0, np.transpose(W)) + vb)
	v_samp_1 = gibbs_sample(v_prob_1)
	all_err  = np.sum((v_samp_1 - x)**2)
	samp_err = all_err/x.shape[0]
	point_err= all_err/(x.shape[0]*x.shape[1])
	return all_err, samp_err, point_err


def softmax_LR_sklearn(train_x, train_y, test_x, test_y):	
	logistic = linear_model.LogisticRegression(penalty='l2', max_iter=300, solver='newton-cg', multi_class='multinomial')
	logistic.fit(train_x, train_y)
	model = logistic
	pred_log_prob = logistic.predict_log_proba(train_x)
	pred_prob     = logistic.predict_proba(train_x)
	decision_res  = logistic.decision_function(train_x)
	pred_res      = logistic.predict(train_x)
	accuracy      = logistic.score(train_x, train_y)
	accuracy      = logistic.score(test_x, test_y)
	test_pred_lab = logistic.predict(test_x)
	print ('test accuracy : ', metrics.accuracy_score(test_y, test_pred_lab))
	print ('test confusion matrix :')
	print (metrics.confusion_matrix(test_y, test_pred_lab, np.unique(train_y)))

def train_by_single_sampel():
	## ##
	n_hidden = 200
	n_sample, n_feature = trainX.shape
	W        = np.zeros((n_feature, n_hidden))
	vb       = np.zeros((n_feature))
	hb       = np.zeros((n_hidden))

	for i in range(n_sample):
		x  = np.array(trainX[i,:].reshape((1, n_feature)))
		grad_W, grad_vb, grad_hb = CD_Grad_one_step(x, W, vb, hb)
		W  = W  + grad_W
		vb = vb + grad_vb
		hb = hb + grad_hb
		if i%1000 == 0:
			all_err, sample_err, point_err = error(trainX, W, vb, hb)
			print (i, 'iter', all_err, '\t', sample_err, '\t', point_err)
			h_prob_0 = sigmoid(np.dot(trainX, W) + hb)
			h_samp_0 = gibbs_sample(h_prob_0)
			print ('hb', hb.shape)
			test_h_prob = sigmoid(np.dot(testX, W) + hb)
			test_h_samp = gibbs_sample(test_h_prob)
			softmax_LR_sklearn(h_samp_0, trainY1, test_h_samp, testY1)
			softmax_LR_sklearn(h_prob_0, trainY1, test_h_prob, testY1)


def train_by_all_data():
	## ##
	n_hidden = 200
	n_sample, n_feature = trainX.shape
	x        = trainX
	W        = np.zeros((n_feature, n_hidden))
	vb       = np.zeros((n_feature))
	hb       = np.zeros((n_hidden))
	print ('hb', hb.shape)

	for i in range(10):
		grad_W, grad_vb, grad_hb = CD_Grad_one_step(x, W, vb, hb)
		W  = W  + grad_W
		vb = vb + grad_vb
		print ('grad_vb', grad_vb.shape)
		print ('grad_hb', grad_hb.shape)
		hb = hb + grad_hb
		print ('hb', hb.shape)
		if i%3 == 0:
			all_err, sample_err, point_err = error(x, W, vb, hb)
			print (i, 'iter', all_err, '\t', sample_err, '\t', point_err)
			h_prob_0 = sigmoid(np.dot(x, W) + hb) 
			print ('h_prob_0', h_prob_0.shape)
			h_samp_0 = gibbs_sample(h_prob_0)
			print ('h_samp_0', h_samp_0.shape)
			print ('testX', testX.shape)
			print ('W', W.shape)
			print ('hb', hb.shape)
			test_h_prob = sigmoid(np.dot(testX, W) + hb)
			print ('test_h_prob', test_h_prob.shape)
			test_h_samp = gibbs_sample(test_h_prob)
			softmax_LR_sklearn(h_samp_0, trainY1, test_h_samp, testY1)
			softmax_LR_sklearn(h_prob_0, trainY1, test_h_prob, testY1)

if __name__=='__main__':
	train_by_all_data()
	#train_by_single_sampel()
