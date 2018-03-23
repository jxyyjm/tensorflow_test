#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tensorflow as tf

iris_data_file = './iris.data'
def get_dataset(file_name, batch_size=10, buffer_size=100, epoch=10):
	# return dataset.get_next() #
	def decode_line(line):
		columns = tf.decode_csv(line, \
				record_defaults=[[0.0], [0.0], [0.0], [0.0], [0]])
		return columns[:-1], columns[-1:]
		## 注意record_defaults会作为默认数据类型去检查field的内容 ##
	dataset = tf.contrib.data.TextLineDataset(file_name)
	dataset = dataset.map(decode_line, num_threads = 5)
	dataset = dataset.shuffle(buffer_size = buffer_size)
	dataset = dataset.repeat(count = epoch)
	dataset = dataset.batch(batch_size = batch_size)
	dataset = dataset.make_one_shot_iterator()
	return dataset

	#x, y = dataset.get_next()
	#y = tf.one_hot(y, 3)
	#print 'dataset y.get_shape', y.get_shape()
	#y = tf.squeeze(y, 1)
	#return x, y

