#!/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf

with tf.Session() as sess:
		with tf.variable_scope('foo') as scope_name:
			x = tf.get_variable('x', initializer=tf.zeros_initializer([1,7]))
			# wrong, if use x = tf.Variable() # not share-variable-type-functioin #
			#x.initializer.run(session=sess) or as following initialize ##
			sess.run(x.initializer)
			print ('x.name : ', x.name)
			scope_name.reuse_variables()
			print ('scope current is : ', tf.get_variable_scope().name)
			y = tf.get_variable('x')
			print ('x is : ', sess.run(x))
			print ('y is : ', sess.run(y))

