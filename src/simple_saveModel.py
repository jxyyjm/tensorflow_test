#!/usr/bin/python
# -*- coding:utf-8 -*-
# @time      : 2017-04-28
# @author    : yujianmin
# @reference :
# @what to-do: save a model and reload it 

## notice :: tf.train.Saver ==> Saves and restores Variables ##
## the Variables could be passed the variables as a dict : tf.train.Saver({'W':W, 'b':b})
##               could be passed               as a list : tf.train.Saver([W, b])
##               could be passed                         : tf.train.Saver({v.op.name: v for v in [W, b]})

## notice :: 1) tf.train.Saver.restore must has define the same Variable; 
##           2) these Variable must be not init pre, because restore is a initialize process; 
##           3) the model path must has the step.

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def save_model_variables():
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	y_ = tf.placeholder('float', [None, 10])
	cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	saver= tf.train.Saver([W, b], max_to_keep=2, keep_checkpoint_every_n_hours=1)

	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
		if i%50 == 0:
			print (i, 'iter','train accuracy', \
				sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
			saver.save(sess, '../data/my-model', global_step=i)
	sess.close()

def reload_model_variables():
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	y_ = tf.placeholder('float', [None, 10])
	cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

	sess = tf.Session()
	saver= tf.train.Saver([W, b]) ## it is initlized the variable ##

	print ('reload ../data/my-model-950')
	saver.restore(sess, tf.train.latest_checkpoint('../data/my-model-950'))

	print (sess.run(b))
	print (sess.run(W))
	print (sess.run(tf.reduce_sum(W)))
	print ('train accuracy', \
		sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#	print ('reload ../data/my-model-50')
#	saver.restore(sess, '../data/my-model-50')
#	print (sess.run(b))
#	print (sess.run(W))
#	print (sess.run(tf.reduce_sum(W)))
#	print (i, 'iter','train accuracy', \
#		sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	sess.close()

if __name__=='__main__':
	save_model_variables()
	reload_model_variables()
	print ('an odd problem, if save and reload function both run, it will error, but single run is all right')
