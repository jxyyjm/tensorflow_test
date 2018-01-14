#!/usr/bin/python
# -*- coding:utf-8 -*-

# @time      : 2017-05-15
# @author    : yujianmin
# @reference : 
# @what to-do: compare SGD and One-line FTL

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
		print (train_x.shape)
		print (train_y.shape)

	def sigmoid(self, x):
		return 1/(1+np.exp(x))
	def bin_SGD(self):
		iris = datasets.load_iris()
		x    = iris.data
		y    = iris.target
		y_0_pos = np.where(y==0)
		y_1_pos = np.where(y==1)
		x_0  = x[y_0_pos]
		x_1  = x[y_1_pos]
		y_0  = y[y_0_pos]
		y_1  = y[y_1_pos]
		data = np.vstack((x_0, x_1))
		label= np.hstack((y_0, y_1))
		train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=100)
		
		n_sample, n_feature = train_x.shape
		n_digits = 10
		theta    = np.zeros([n_feature])
		grad     = np.zeros([n_feature])
		lr       = 0.02
		
		for i in range(train_x.shape[0]):
			cur_sample = train_x[i]
			cur_label  = train_y[i]
			for i in range(len(theta)):
				prod_   = np.dot(cur_sample, theta)
				pred_cur= self.sigmoid(-prod_)
				for i in range(len(grad)):
					grad[i] = (pred_cur - cur_label)*cur_sample[i] # here and next line is same #
					#grad[i] = -cur_label*cur_sample[i]*self.sigmoid(prod_) + (1-cur_label)*cur_sample[i]*self.sigmoid(-prod_)
					theta[i]= theta[i] - lr*grad[i]
			# train accuracy #
			pred_train  = np.dot(train_x, theta)
			pred_0_pos  = np.where(pred_train<0.5)
			pred_1_pos  = np.where(pred_train>=0.5)
			pred_label  = pred_train
			pred_label[pred_0_pos] = 0
			pred_label[pred_1_pos] = 1
			train_accuracy = metrics.accuracy_score(train_y, pred_label)
			# test  accuracy #
			pred_test  = np.dot(test_x, theta)
			pred_0_pos  = np.where(pred_test<0.5)
			pred_1_pos  = np.where(pred_test>=0.5)
			pred_label  = pred_test
			pred_label[pred_0_pos] = 0
			pred_label[pred_1_pos] = 1
			test_accuracy = metrics.accuracy_score(test_y, pred_label)
			print ('train_accuracy is:' + str(train_accuracy) +'\t'+ 'test_accuracy is:' + str(test_accuracy))
				


	def bin_FTRL(self):
		pass
		iris = datasets.load_iris()
		x    = iris.data
		y    = iris.target
		y_0_pos = np.where(y==0)
		y_1_pos = np.where(y==1)
		x_0  = x[y_0_pos]
		x_1  = x[y_1_pos]
		y_0  = y[y_0_pos]
		y_1  = y[y_1_pos]
		data = np.vstack((x_0, x_1))
		label= np.hstack((y_0, y_1))
		train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=100)
		# parameter could set #
		alpha = 0.01
		beta  = 0.02
		lambda1 = 0.5
		lambda2 = 2
		# mid-para #
		n_sample, n_feature = x.shape
		theta = np.zeros([n_feature])
		z     = np.zeros([n_feature])
		n     = np.zeros([n_feature])
		grad  = np.zeros([n_feature])
		sigma = np.zeros([n_feature])

		for i in range(train_x.shape[0]):
			cur_sample = train_x[i]
			cur_label  = train_y[i]
			for i in range(len(theta)):
				if abs(z[i])<= lambda1: theta[i]=0
				else: theta[i] = - 1/((beta + np.sqrt(n[i]))/alpha + lambda2) * (z[i] - lambda1*np.sign(z[i]))
			pred_cur = self.sigmoid(np.dot(cur_sample, theta))
			for i in range(len(grad)):
				grad[i] = (pred_cur - cur_label)*cur_sample[i]
				sigma[i]= (np.sqrt(n[i] + grad[i]**2) - np.sqrt(n[i]))/alpha
				z[i]    = z[i] + grad[i] - sigma[i]*theta[i]
				n[i]    = n[i] + grad[i]**2
			# train accuracy #
			pred_train  = np.dot(train_x, theta)
			pred_0_pos  = np.where(pred_train<0.5)
			pred_1_pos  = np.where(pred_train>=0.5)
			pred_label  = pred_train
			pred_label[pred_0_pos] = 0
			pred_label[pred_1_pos] = 1
			train_accuracy = metrics.accuracy_score(train_y, pred_label)
			# test  accuracy #
			pred_test  = np.dot(test_x, theta)
			pred_0_pos  = np.where(pred_test<0.5)
			pred_1_pos  = np.where(pred_test>=0.5)
			pred_label  = pred_test
			pred_label[pred_0_pos] = 0
			pred_label[pred_1_pos] = 1
			test_accuracy = metrics.accuracy_score(test_y, pred_label)
			print ('train_accuracy is:' + str(train_accuracy) +'\t'+ 'test_accuracy is:' + str(test_accuracy))
		## what a happy life, if repet the train-data, it will get 100% ##
		'''
		for i in range(train_x.shape[0]):
			cur_sample = train_x[i]
			cur_label  = train_y[i]
			for i in range(len(theta)):
				if abs(z[i])<= lambda1: theta[i]=0
				else: theta[i] = - 1/((beta + np.sqrt(n[i]))/alpha + lambda2) * (z[i] - lambda1*np.sign(z[i]))
			pred_cur = self.sigmoid(np.dot(cur_sample, theta))
			for i in range(len(grad)):
				grad[i] = (pred_cur - cur_label)*cur_sample[i]
				sigma[i]= (np.sqrt(n[i] + grad[i]**2) - np.sqrt(n[i]))/alpha
				z[i]    = z[i] + grad[i] - sigma[i]*theta[i]
				n[i]    = n[i] + grad[i]**2
			# train accuracy #
			pred_train  = np.dot(train_x, theta)
			pred_0_pos  = np.where(pred_train<0.5)
			pred_1_pos  = np.where(pred_train>=0.5)
			pred_label  = pred_train
			pred_label[pred_0_pos] = 0
			pred_label[pred_1_pos] = 1
			train_accuracy = metrics.accuracy_score(train_y, pred_label)
			# test  accuracy #
			pred_test  = np.dot(test_x, theta)
			pred_0_pos  = np.where(pred_test<0.5)
			pred_1_pos  = np.where(pred_test>=0.5)
			pred_label  = pred_test
			pred_label[pred_0_pos] = 0
			pred_label[pred_1_pos] = 1
			test_accuracy = metrics.accuracy_score(test_y, pred_label)
			print ('train_accuracy is:' + str(train_accuracy) +'\t'+ 'test_accuracy is:' + str(test_accuracy))
		'''
	def mult_FTRL(self):
		pass
	def model_save(self):
		pass
if __name__=='__main__':
	model = CSimple_test()
	model.read_data_split()
	#model.softmax_SGD()
	#print ('last 0.91 is the highest est-accuracy')
	model.bin_FTRL()
	print ('########')
	model.bin_SGD()
