#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

## input and output ##
x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

## weight and bias ##
w  = tf.Variable(tf.zeros(shape = [784, 10]))
b  = tf.Variable(tf.zeros(shape = [10])) ## y = w*x + b ## 10-output ## b.size=[10]

## target-func and loss-func ##
y  = tf.matmul(x, w) + b
cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

## initialize variables ##
sess.run(tf.initialize_all_variables())

## train ##
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entroy)
for i in range(2000):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={ x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


