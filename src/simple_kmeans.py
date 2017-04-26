#!/usr/bin/python

## @time       : 2017-04-26
## @author     : yujianmin
## @reference  : 
## @what-to-do : try using tensorflow to make a test-cluster class

from __future__ import division
from __future__ import print_function

import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn import datasets
from numpy import linalg as la
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
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
		self.kmeans  = ''
		self.cen2lab = ''
	def __del__(self):
		self.train_x = ''
		self.train_y = ''
		self.test_x  = ''
		self.test_y  = ''
		self.kmeans  = ''
		self.cen2lab = ''
	def read_data_split(self):
		#iris = datasets.load_iris()
		digits = datasets.load_digits()
		x    = digits.data
		y    = digits.target
		#data = np.hstack((x, y.reshape((y.shape[0],1))) ## merge
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=50)
		self.train_x = train_x
		self.train_y = train_y
		self.test_x  = test_x
		self.test_y  = test_y
	
	def map_center2label_similarity(self):
		cen2lab = {}
		# compute the real-center map real-label #
		digits  = np.unique(self.train_y)
		for i in digits:
			pos = np.where(self.train_y==i)
			center = np.mean(self.train_x[pos])
			cen2lab[i] = center
		# map pred-center to real-label #
		pred_centers   = self.kmeans.cluster_centers_
		n_row, n_col   = pred_centers.shape
		for i in range(n_row):
			pred_center= pred_centers[i,:]
			min_dist   = np.float('inf')
			map_label  = np.float('-inf')
			for label, real_center in cen2lab.iteritems():
				dist   = la.norm(real_center-pred_center)
				message= str(i) +'\t'+ str(label) +'\t'+ str(dist)
				logging.debug(message)
				if float(dist) < float(min_dist):
					map_label = label
					min_dist  = dist
			cen2lab[i] = map_label
			message    = 'last, ' + str(i) + '===>' + str(map_label)
			logging.debug(message)
		self.cen2lab   = cen2lab
		## this method to map, is difficult because the similarity is hard to find ##
	
	def map_center2label_most(self):
		## which is most, and this center is mapped to which num ##
		cen2lab      = {}
		train_y_pred = self.kmeans.predict(self.train_x)
		pre_digts    = np.unique(train_y_pred)
		if len(pre_digts) != 10:
			print ('not k-means your expect 10-cluster, please re-run')
			sys.exit()
		message      = 'pre_digts:' + str(pre_digts)
		logging.debug(message + 'pred len : '+str(len(train_y_pred)) +'real len : '+str(len(self.train_y)))
		for i in range(10):
			pos_pred = np.where(train_y_pred==i)
			real_y   = self.train_y[pos_pred]
			message  = 'cur_real_y : ' + str(real_y)
			logging.debug(message)
			stat_num = np.bincount(real_y)
			stat_num = [x for x in stat_num if x>0]
			real_lab = np.unique(real_y)
			message  = 'stat_num : ' + str(stat_num)
			logging.debug(message)
			message  = 'real_lab : ' + str(real_lab)
			logging.debug(message)
			max_pos  = np.argmax(stat_num)
			map_lab  = real_lab[max_pos]
			cen2lab[i] = map_lab
			message    = 'last, ' + str(i) + '===>' + str(map_lab)
			logging.debug(message)
		self.cen2lab   = cen2lab
		## this method to map, is more reliable ##

	def k_means_sklearn(self, simi='cos'):
		train_x = scale(self.train_x, axis=0)
		# scale(X, axis=0, with_mean=True, with_std=True, copy=True)
		# axis = 0 standardization by col #
		train_y = self.train_y
		n_sample, n_feature = train_x.shape
		n_digits= len(np.unique(train_y))
		message = 'n_sample:' +'\t'+ str(n_sample) +'\t'+ 'n_feature:' +'\t'+ str(n_feature) +'\t'+ 'labels:' +'\t'+ str(n_digits)
		logging.debug(message)
		sklearn_kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=n_digits, max_iter=500)
		sklearn_kmeans.fit(X=train_x)
		self.kmeans    = sklearn_kmeans
		#self.map_center2label_similarity()
		logging.debug(sklearn_kmeans.get_params)
		logging.debug(sklearn_kmeans.cluster_centers_.shape)
		train_y_pred = sklearn_kmeans.predict(X=train_x)
		
		self.map_center2label_most()
		train_y_pred_= []
		for i in train_y_pred:
			map_label= self.cen2lab[i]
			train_y_pred_.append(map_label)
		train_y_pred = np.array(train_y_pred_, dtype=np.int32)
		print ('train accuracy : ', metrics.accuracy_score(train_y, train_y_pred))
		print ('train confusion matrix:')
		print (metrics.confusion_matrix(train_y, train_y_pred))
		test_y_pred  = sklearn_kmeans.predict(X=scale(self.test_x, axis=0))
		test_y_pred_ = []
		for i in test_y_pred:
			map_label= self.cen2lab[i]
			test_y_pred_.append(map_label)
		test_y_pred  = np.array(test_y_pred_, dtype=np.int32)
		print ('test accuracy : ', metrics.accuracy_score(self.test_y, test_y_pred))
		print ('test confusion matrix:')
		print (metrics.confusion_matrix(self.test_y, test_y_pred))
	def k_means_tf(self, simi='cos'):
		pass
		## input.placeholder ##
		n_sample, n_feature = self.train_x.shape
		n_digits = len(np.unique(self.train_y))
		x = tf.placeholder(tf.float32, [n_sample, n_feature])
		y = tf.placeholder(tf.float32, [n_sample, 1])
		## para.variable  ##
		center = tf.Variable(tf.float32, [n_digits, n_feature])
		sign   = tf.Variable(tf.float32, [n_sample, 1]
		## compute min   ##
		
		## initilize ##
		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		## train ##
		for i range(300):
			if i == 0:
				tf.
	def model_save(self):
		pass
if __name__=='__main__':
	CTest = CSimple_test()
	## read the data ##
	CTest.read_data_split()
	## train a cluster-model using sklearn ##
	CTest.k_means_sklearn()

