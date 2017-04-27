#!/usr/bin/python
# -*- coding:utf-8 -*-

# @time      : 2017-04-27
# @author    : yujianmin
# @reference : 
# @what to-do: try a simple tensorflow model

import tensorflow as tf
import numpy as np

## data      ##
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300

## parameter ##
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

## 目标与优化 ##
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

## start ##
sess = tf.Session()
sess.run(init)

## 拟合 ##
for step in xrange(0, 201):
	sess.run(train)
	if step % 20 == 0:
		print step, sess.run(W), sess.run(b)
