#!~/anaconda2/bin/python
# -*- coding:utf-8 -*-

# @time       : 2017-02-19
# @author     : yujianmin
# @reference  : 
# @what to-do : try to read sparse-input-data, and build a model 

from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import roc_auc_score

logging.basicConfig(
	level   = logging.DEBUG,
	format  = '%(asctime)s %(filename)s[line:%(lineno)d] %(funcName)s %(levelname)s %(message)s',
	datefmt = '%Y-%m-%d %H:%M:%S'
)

class Class_LogisticRegression():
	def __init__(self):
		self.file_input = ''
		self.train_X    = ''
		self.train_Y    = ''
		self.test_X     = ''
		self.test_Y     = ''
		self.model      = ''
		self.L_rate     = 0.05
		self.iter_max   = 30000
	def load_sparse_data(self, file_input=''):
		'''
		文件内容格式 label pos:value pos:value pos:value
		par input : filepath is a file-name
	     			line type is "label 1:pos1 1:pos14 1:pos30"
		par output: X :: tf.SparseTensor, feature-value
		par output: Y :: tf.Tensor, label-value
		'''
		if file_input == '': filepath = self.file_input
		else:                filepath = file_input
		if os.path.isfile(filepath) == False:
			logging.error('no data file named :'+filepath); return -1, -1

		handle = open(filepath, 'r')
		row, indices, values, labels = 0, [], [], []
		for line in handle:
			line  = line.strip()
			if len(line) <= 0:
				logging.warning('len(line) < 0'); continue
			words = line.split(' ')
			if len(words) < 2:
				logging.warning('len words < 2, continue'); continue
			y_    = int(words[0].strip())
			label = [y_]
			value_list  = words[1:]
			
			cur_indices = [[row, int(i.split(':')[0])] for i in value_list]
			cur_value   = [int(i.split(':')[1]) for i in value_list]
			indices += cur_indices
			values  += cur_value
			labels  += label
			row += 1
			if row%100 == 0:
				logging.debug('current row is : ' + str(row))
		handle.close()
	
		x_res     = tf.SparseTensor(indices, values, dense_shape=[row, 30000]) ## 这里用 tf.SparseTensor 构造了稀疏的数据输入 ##
		y_res = tf.constant(labels, dtype=tf.int32, shape=[row, 1])
		logging.debug('in class Cread_data::load_sparse_data transed example x is : '+ str(x_res))
		logging.debug('in class Cread_data::load_sparse_data transed example y is : '+ str(y_res))
		return x_res, y_res 
		
	def train(self, trainX, trainY):
		sess  = tf.Session()
		print ('sparse train_x shape :', sess.run(trainX).dense_shape)
		print ('train_y shape :', sess.run(trainY).shape)
		feat_num = sess.run(trainX).dense_shape[1]
		logging.debug('feat_num is : '+str(feat_num))
		input_x  = tf.sparse_placeholder(dtype = tf.float32, shape = None) ## notice :: here is None ##
		output_y = tf.placeholder(dtype = tf.float32, shape = [None, 1])
		theta    = tf.Variable(tf.random_uniform(shape  = [feat_num, 1], minval = 0, maxval = 1))
		bias     = tf.Variable(tf.zeros([1]))
		pre_y    = tf.sparse_tensor_dense_matmul(input_x, theta) + bias
		loss     = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre_y, labels=output_y))
		## here tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=real)
		## ==> -real*log[sigmod(pred)] - (1-real)*log[1-sigmod(pred)] 
		L_rate   = self.L_rate
		optimizer= tf.train.GradientDescentOptimizer(L_rate)
		train    = optimizer.minimize(loss)
		predict  = tf.nn.sigmoid(pre_y)
		init  = tf.initialize_all_variables()
		sess.run(init)
		max_steps= self.iter_max
		for step in xrange(max_steps):
			sess.run(train, feed_dict = {input_x : sess.run(X), output_y : sess.run(Y)})
			if step%100 == 0:
				cur_pre, cur_cost = sess.run([predict, loss], feed_dict = {input_x: sess.run(X), output_y: sess.run(Y)})
				print (step, 'cur train auc : ', roc_auc_score(sess.run(Y), cur_pre), 'cur train loss : ', cur_cost)
	
if __name__=='__main__':
	model = Class_LogisticRegression()
	X, Y  = model.load_sparse_data('../data/data.binLabel.part3')
	model.train(X, Y)

