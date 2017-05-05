#!/usr/bin/python
# -*- coding:utf-8 -*-

# @time      : 2017-05--4
# @author    : yujianmin
# @reference : An Introduction to Restricted Boltzmann Machine
# @what to-do: try to build a RBM model ## RBM could get a distribution from training data ##


from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn import datasets
try: from sklearn.cross_validation import train_test_split
except: from sklearn.model_selection import train_test_split

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
class CTest_RBM:
	def __init__(self, hidden_nodes):
		self.W = ''
		self.b = '' # input-layer  bias
		self.c = '' # hidden-layer bias
		self.trainX = ''
		self.trainY = ''
		self.testX  = ''
		self.testY  = ''
		self.input_nodes = ''
		self.hidden_nodes= hidden_nodes

	def __del__(self):
		self.W = ''
		self.b = ''
		self.c = ''
		self.trainX = ''
		self.trainY = ''
		self.testX  = ''
		self.testY  = ''

	def read_data(self):
		digits = datasets.load_digits()
		x      = digits.data
		y      = digits.target
		trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=100)
		self.trainX = trainX
		self.trainY = trainY
		self.testX  = testX
		self.testY  = testY

	def gibbs_sample_h_given_v(self, v, prob):
		gibbs_sample= np.random.binomial(1, prob, [len(self.b), 1])
		return gibbs_sample

	def gibbs_sample_v_given_h(self, h, prob):
		gibbs_sample= np.random.binomial(1, prob, [len(self.c), 1])
		return gibbs_sample

	def train_by_single_data(self):
		## ##
		print ('self.trainX[0,:]')
		print (self.trainX[0,:])

		n_sample, n_feature = self.trainX.shape
		v      = tf.placeholder(shape=[None, n_feature], dtype=tf.float32)
		W_     = tf.Variable(tf.zeros([self.hidden_nodes, n_feature], dtype=tf.float32))
		b_     = tf.Variable(tf.zeros([n_feature, 1], dtype=tf.float32))
		c_     = tf.Variable(tf.zeros([self.hidden_nodes,1], dtype=tf.float32))
		prob_h = tf.nn.sigmoid(tf.matmul(self.W, v) + self.c)
		h_samp = self.gibbs_sample_h_given_v(v, prob_h)
		prob_v = tf.nn.sigmoid(tf.matmul(self.W, h_samp) + self.b)
		v_samp = self.gibbs_sample_v_given_h(h_samp, prob_v)
		grad_W = tf.nn.matmul(prob_b, v) - tf.nn.matmul(prob_v, v_samp)
		grad_b = v - v_sampe
		grad_c = prob_h - prob_v
		
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)
		for i in range(n_sample):
			sess.run(grad_W, feed_dict={v: self.trainX[i, :]})
			sess.run(grad_b, feed_dict={v: self.trainX[i, :]})
			sess.run(grad_c, feed_dict={v: self.trainX[i, :]})
			sess.run(W_.assign_add(grad_W))
			sess.run(b_.assign_add(grad_b))
			sess.run(c_.assign_add(grad_c))
		self.W = W_
		self.b = b_
		self.c = c_
		print ('last running: self.W')
		print (sess.run(self.W))
		print ('last running: self.b')
		print (sess.run(self.b))
		print ('last running: self.c')
		print (sess.run(self.c))

	def error_compute(self):
		## 
		h_prob  = tf.nn.sigmoid(tf.matmul(self.W, self.trainX) + self.c)
		h_value = self.gibbs_sample_h_given_v(self.trainX, h_prob)
		v_pred  = tf.nn.sigmoid(tf.matmul(self.W, h_value) + self.b)
		v_value = self.gibbs_sample_v_given_h(h_value, v_pred)
		train_error = tf.reduce_mean((v_value-self.trainX)**2)
		print ('train error: ', sess.run(train_error))
		
		
if __name__=='__main__':
	CTest = CTest_RBM(100)
	CTest.read_data()
	CTest.train_by_single_data()
	CTest.error_compute()

