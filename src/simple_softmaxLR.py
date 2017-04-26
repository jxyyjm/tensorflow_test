#!/usr/bin/python

## @time       : 2017-04-26
## @author     : yujianmin
## @reference  : 
## @what-to-do : try using tensorflow to make a test-LR class

from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split


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
	def __del__(self):
		self.train_x = ''
		self.train_y = ''
		self.test_x  = ''
		self.test_y  = ''
		self.model   = ''
	def read_data_split(self):
		#iris = datasets.load_iris()
		digits = datasets.load_digits()
		x    = digits.data
		y    = digits.target
		#data = np.hstack((x, y.reshape((y.shape[0],1))) ## merge
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=150)
		self.train_x = train_x
		self.train_y = train_y
		self.test_x  = test_x
		self.test_y  = test_y

	def softmax_LR_tf(self):
		pass
		## input.placeholder ##
		n_sample, n_feature = self.train_x.shape
		n_digits = len(np.unique(self.train_y))
		x = tf.placeholder(tf.float32, [n_sample, n_feature])
		y = tf.placeholder(tf.float32, [n_sample, 1])
		## para.variable  ##
		w = tf.Variable(tf.zeros([n_feature, n_digits]))
		## initilize ##
		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		## train ##
		for i range(300):
				pass
	def softmax_LR_epoch_tf(self):
		pass
	def model_save(self):
		pass
if __name__=='__main__':
	CTest = CSimple_test()
	## read the data ##
	CTest.read_data_split()
	## train a cluster-model using sklearn ##
	CTest.softmax_LR_tf()

