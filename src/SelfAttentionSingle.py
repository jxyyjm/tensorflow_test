#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys
import six
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
  ## 测试self-attention Matrix Compute ##
'''
import tensorflow as tf
import copy

## [seq_length, emb_size] ##
query = tf.Variable(tf.constant( \
    [[1.1, 4.2, 1.3, 1.4], \
     [0.1, 2.3, 2.5, 2.7], \
     [3.1, 3.4, 3.8, 3.0]],\
    dtype=tf.float32), name='query')
key   = tf.Variable(tf.constant( \
    [[1.1, 4.2, 1.3, 1.4], \
     [0.1, 2.3, 2.5, 2.7], \
     [3.1, 3.4, 3.8, 3.0]],\
    dtype=tf.float32), name='key')
value = tf.Variable(tf.constant( \
    [[1.1, 4.2, 1.3, 1.4], \
     [0.1, 2.3, 2.5, 2.7], \
     [3.1, 3.4, 3.8, 3.0]],\
    dtype=tf.float32), name='value')

gpu_options = tf.GPUOptions(allow_growth = True)
with tf.Session(config = tf.ConfigProto( \
                gpu_options = gpu_options, \
                allow_soft_placement = True, \
                log_device_placement = False)) as sess:
  sess.run(tf.global_variables_initializer())
  print '='*20, 'query'
  print sess.run(query)
  print '='*20, 'key'
  print sess.run(key)
  print '='*20, 'value'
  print sess.run(value)
  print '='*20, 'query*key'
  z = tf.matmul(query, key, transpose_b=True)
  print sess.run(z)
  scaled_z = tf.multiply(z, 1/tf.sqrt(4.0))
  print '='*20, 'query*key*scaled'
  print sess.run(scaled_z)
  print '='*20, 'softmax(query*key*scaled)'
  softmax_z= tf.nn.softmax(scaled_z, dim=1)
  print sess.run(softmax_z)

  print '='*20, 'query*key x value', '='*10, ' method-1'
  print sess.run(tf.tensordot(softmax_z, value, [[1],[0]])) ## 沿着 z的列，value的行 做点乘dot ##

  print '='*20, 'query*key x value', '='*10, ' method-2'
  print sess.run(tf.tensordot(softmax_z, value, 1))

  print '='*20, 'query*key x value', '='*10, ' method-3'
  print sess.run(tf.matmul(softmax_z, value))

