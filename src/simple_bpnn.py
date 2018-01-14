#!/usr/bin/python
# -*- coding:utf-8 -*

## @time       : 2017-06-17
## @author     : yujianmin
## @reference  : http://blog.csdn.net/yujianmin1990/article/details/49935007
## @what-to-do : try to make a any-layer-nn by hand (one-input-layer; any-hidden-layer; one-output-layer)

from __future__ import division
from __future__ import print_function
import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf
from  tensorflow import metrics 
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100
num_epochs = 10
MAX_ITER_NUM = int(60000/batch_size * num_epochs)
## =========================================== ##
## =========================================== ##


def read_data(filename):
	mnist = input_data.read_data_sets(filename, one_hot=True)
	train_data = mnist.train
	test_data  = mnist.test
	return train_data, test_data
    
def main(train_data_path):
	#print ('debug# FLAGS', FLAGS) 
	train_data, test_data = read_data(train_data_path)
	
	x  = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
	y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
	W1 = tf.Variable(tf.random_normal([784, 128]))
	W2 = tf.Variable(tf.random_normal([128, 10]))
	b1 = tf.Variable(tf.zeros([128]))
	b2 = tf.Variable(tf.zeros([10]))
	
	y			 = tf.nn.softmax(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1)), W2),b2))
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	grad_op	   = tf.train.GradientDescentOptimizer(learning_rate=0.5)
	train_op	  = grad_op.minimize(cross_entropy)
	
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
	accuracy_op		= tf.reduce_mean(tf.cast(correct_prediction, 'float'))
		
	init_op	= tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init_op)
		for i in range(MAX_ITER_NUM):
			train_x, train_y = train_data.next_batch(batch_size)
			loss, _ = sess.run([cross_entropy, train_op], feed_dict={x: train_x, y_: train_y})
			if i%10 == 0:
				accuracy = sess.run(accuracy_op, feed_dict={x: test_data.images, y_: test_data.labels})
				print ('iter : ', i, 'loss : ', loss, 'test accuracy : ', accuracy)

if __name__=='__main__':
	time_begin = time.ctime()
	train_data_path = '../data/MNIST_data' 
	main(train_data_path)
	time_end   = time.ctime()
	print ('time begin : ', time_begin)
	print ('time  end  : ', time_end)
