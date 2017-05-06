#!/usr/bin/python

## @time       : 2017-04-26
## @author     : yujianmin
## @reference  : code,       , https://www.tensorflow.org/get_started/mnist/beginners
##             : cross-entroy, https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/chap3/c3s1.html
## @what-to-do : try using tensorflow to make a multi-class predict model

from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

logging.basicConfig(
	level   = logging.DEBUG,
	format  = '%(asctime)s %(filename)s[line:%(lineno)d] %(funcName)s %(levelname)s %(message)s',
	datefmt = '%Y-%m-%d %H:%M:%S',
	filename= './tmp.log',
	filemode= 'w'
	)


class CSimple_test:
	def __init__(self):
		self.train_x = ''
		self.train_y = ''
		self.test_x  = ''
		self.test_y  = ''
		self.model   = ''
		self.data    = ''
	def __del__(self):
		self.train_x = ''
		self.train_y = ''
		self.test_x  = ''
		self.test_y  = ''
		self.model   = ''
		self.data    = ''

	def read_data_split(self):
		#iris = datasets.load_iris()
		#digits = datasets.load_digits()
		#x    = digits.data
		#y    = digits.target
		#data = np.hstack((x, y.reshape((y.shape[0],1))) ## merge
		#train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=150)
		mnist   = input_data.read_data_sets('MNIST_data/', one_hot=True)
		train_x = mnist.train.images
		train_y = mnist.train.labels
		test_x  = mnist.test.images
		test_y  = mnist.test.labels
		self.train_x = train_x
		self.train_y = train_y
		self.test_x  = test_x
		self.test_y  = test_y
		self.data    = mnist

	def read_data_split_one(self):
		# made the input as 0/1 value #
		mnist   = input_data.read_data_sets('MNIST_data/', one_hot=True)
		train_x = mnist.train.images
		one_pos = np.where(train_x>0)
		train_x[one_pos] = 1
		train_y = mnist.train.labels
		
		test_x  = mnist.test.images
		one_pos = np.where(test_x>0)
		test_x[one_pos] = 1
		test_y  = mnist.test.labels
		self.train_x = train_x
		self.train_y = train_y
		self.test_x  = test_x
		self.test_y  = test_y
		self.data    = mnist

	def softmax_tf(self):
		## it could seen combine 10-small-models into One-big-model and optimal all-para sync ##
		## input.placeholder ##
		n_sample, n_feature = self.train_x.shape
		n_digits = 10
		x = tf.placeholder(tf.float32, [None, n_feature])
		y = tf.placeholder(tf.float32, [None, n_digits])
		## para.variable  ##
		w = tf.Variable(tf.zeros([n_feature, n_digits]))
		b = tf.Variable(tf.zeros([n_digits])) # each small-model has its bias #
		## object-function##
		'''
			< tf.nn.softmax-family >
			@@softmax :
				==> softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))
				==> softmax[i, j] = exp(j)/sum_j(exp([i,j]))
			@@log_softmax
			@@sigmoid_cross_entropy_with_logits(logits=X, labels=Y)
                ==> -Y*log[sigmod(X)] - (1-Y)*log[1-sigmod(X)] ## is also used as loss-function
			@@softmax_cross_entropy_with_logits
			@@sparse_softmax_cross_entropy_with_logits
			@@weighted_cross_entropy_with_logits
		'''
		f_fun = tf.nn.softmax(tf.matmul(x, w) + b) ## [m, 10]
		## loss-function   ## cross_entropy ##
		#loss  = tf.reduce_mean(-tf.reduce_sum(y * tf.log(f_fun))) ## here is not right, tf.reduce_sum need specify dim ##
		loss  = tf.reduce_mean(-tf.reduce_sum(y * tf.log(f_fun), 1))
		#loss  = tf.reduce_mean(-tf.reduce_sum(y * tf.log(f_fun)) - 0.01*tf.reduce_sum(b))
		## cross_entropy = -1/n sum_x [y*lnf(x) + (1-y)*lnf(x)] ##
		## optimizer       ##
		'''
			@@Optimizer    
			@@GradientDescentOptimizer
			@@AdadeltaOptimizer
			@@AdagradOptimizer
			@@MomentumOptimizer
			@@AdamOptimizer
			@@FtrlOptimizer
			@@RMSPropOptimizer
		'''
		## build a gradient descent optimizer here ##
		lr   = 0.005
		step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
		## tf.train.GradientDescentOptimizer return==> a optimizer
		## optimizer.minize(loss) equal ==> compute_gradients() then, apply_gradient
		## evaluate        ##
		equal_bool = tf.equal(tf.argmax(y, 1), tf.argmax(f_fun, 1))
		accuracy   = tf.reduce_mean(tf.cast(equal_bool, tf.float32))
		## initilize       ##
		#init = tf.initialize_all_variables()
		init = tf.global_variables_initializer()
		sess = tf.Session()
		#sess = tf_debug.LocalCLIDebugWrapperSession()
		sess.run(init)
		## train           ##
		for i in range(500):
				sess.run(step, feed_dict={x: self.train_x, y: self.train_y})
				if i%10 == 0:
					print ('train accuracy : ', \
							sess.run(accuracy, feed_dict={x:self.train_x, y:self.train_y}), \
							'test accuracy : ', \
							sess.run(accuracy, feed_dict={x:self.test_x, y:self.test_y})
						  )
					
	def softmax_epoch_tf(self):
		n_sample, n_feature = self.train_x.shape
		n_digits = 10
		x = tf.placeholder(tf.float32, [None, n_feature])
		y = tf.placeholder(tf.float32, [None, n_digits])
		w = tf.Variable(tf.zeros([n_feature, n_digits]))
		b = tf.Variable(tf.zeros([n_digits]))
		f_fun = tf.nn.softmax(tf.matmul(x, w) + b) ## [m, 10]
		#loss  = tf.reduce_mean(-tf.reduce_sum(y * tf.log(f_fun))) # here is also wrong #
		loss  = tf.reduce_mean(-tf.reduce_sum(y * tf.log(f_fun), 1)) #+ 0.1 * tf.reduce_sum(w*w)
		## if regularity is added ##
		#loss  = tf.reduce_mean(-tf.reduce_sum(y * tf.log(f_fun)) - 0.01*tf.reduce_sum(b))
		lr   = 0.008
		#step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
		step = tf.train.AdamOptimizer(lr).minimize(loss)
		equal_bool = tf.equal(tf.argmax(y, 1), tf.argmax(f_fun, 1))
		accuracy   = tf.reduce_mean(tf.cast(equal_bool, tf.float32))
		#init = tf.initialize_all_variables()
		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		for i in range(2500):
				batch_x, batch_y = self.data.train.next_batch(100)
				sess.run(step, feed_dict={x: batch_x, y: batch_y})
				if i%10 == 0:
					print ('train accuracy : ', \
							sess.run(accuracy, feed_dict={x:self.train_x, y:self.train_y}), \
							'test accuracy : ', \
							sess.run(accuracy, feed_dict={x:self.test_x, y:self.test_y})
						  )

	def model_save(self):
		pass
	
if __name__=='__main__':
	CTest = CSimple_test()
	
	## read the data ##
	CTest.read_data_split()
	# train a softmax multi-classes model
	print ('train using all-data')	
	#CTest.softmax_tf()
	print ('train using batch-data')
	CTest.softmax_epoch_tf()
	print ('if using all-data, learning-rate 0.000001, if mini-batch-data, lr 0.001')
	print ('mini-batch, conv untill 0.92 around, if could it be more higher ?')
	print ('sklearn softmax 0.92 around')
	'''
	prnit ('input value set 0/1 format')
	CTest.read_data_split_one()
	CTest.softmax_tf()
	'''
