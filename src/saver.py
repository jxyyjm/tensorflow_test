#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tensorflow as tf

iris_data_file = './iris.data'
def get_batch_data(file_name, batch_size=10, buffer_size=100, epoch=10):
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
	x, y = dataset.get_next()
	y = tf.one_hot(y, 3)
	#print 'dataset y.get_shape', y.get_shape()
	y = tf.squeeze(y, 1)
	return x, y
	
with tf.variable_scope('layer-1'):
	w1 = tf.Variable( \
			tf.random_normal(shape = [4,8], mean  = 0.0, stddev= 1.0),\
			dtype = tf.float32, \
			name  = 'w1')
	b1 = tf.Variable(tf.zeros([1,8]), dtype = tf.float32, name = 'b1')
with tf.variable_scope('layer-2'):
	w2 = tf.Variable( \
			tf.random_normal(shape = [8,3], mean  = 0.0, stddev= 1.0),\
			dtype = tf.float32, \
			name  = 'w2')
	b2 = tf.Variable(tf.zeros([1,3]), dtype = tf.float32, name = 'b2')

x = tf.placeholder(dtype = tf.float32, shape=[None, 4], name = 'x')
y = tf.placeholder(dtype = tf.int32,   shape=[None, 3], name = 'y')
y1= tf.matmul(x, w1) + b1
y2= tf.matmul(y1,w2) + b2
y3= tf.nn.softmax(y2, 1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y2))

labels      = tf.argmax(y, 1)
predictions = tf.argmax(y3, 1)
accuracy    = tf.contrib.metrics.accuracy(labels = labels, predictions = predictions)
confusion   = tf.contrib.metrics.confusion_matrix(labels = labels, predictions = predictions)
train_op    = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
print ('train_op', train_op)
next_data   = get_batch_data(iris_data_file)
#cur_saver   = tf.train.Saver(max_to_keep=4)#, keep_checkpoint_every_n_hours=1)
cur_saver   = tf.train.Saver([w1, b1, w2, b2], max_to_keep=4)
tf.add_to_collection("loss", loss)
tf.add_to_collection('predictions', predictions)
tf.add_to_collection('accuracy', accuracy)
tf.add_to_collection('confusion', confusion)
tf.add_to_collection('train_op', train_op)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	try:
		iter = 0
		while True:	
			input, label = sess.run(next_data)
			#print 'input.shape', input.shape, 'label.shape', label.shape
			k, l, p = sess.run([y3, labels, predictions], feed_dict={x:input, y:label})
			#print 'label.shape=', l.shape, 'prediction.shape=', p.shape
			sess.run(train_op, feed_dict={x:input, y:label})
			if iter % 50 == 1:
				train_loss, train_accuracy, train_confusion \
					= sess.run( \
						[loss, accuracy, confusion], \
						feed_dict={x:input, y:label})
				print ('iter:', iter, \
						'train_loss:', train_loss, \
						'train_accuracy:', train_accuracy, \
						'train_confusion:', train_confusion)
				save_path = cur_saver.save(sess, "./tmp/my-model.ckpt", \
											global_step = iter)
			iter += 1
	except tf.errors.OutOfRangeError:
		print 'end'
	print ('save_path', save_path)

