#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys 
import six 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

'''
  ## 测试self-attention Batch Matrix Compute ##
  ## 1) 对tf.matmul([batch, x, y], [batch, z, y], transpose_b=True) 而言，           ##
  ##    会使用gen_math_ops.batch_mat_mul(), 自动将 [0, 1, 2] --> [0, 2, 1]           ##
  ##    输出 output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])         ##
  ##    而使用tf.transpose([batch, z, y]) --> [y, z, batch]; [0, 1, 2] --> [2, 1, 0] ##
  ##    tf.matmul() 永远只计算最后两维表示的矩阵，前面的维度统统保存下来             ##
'''
import tensorflow as tf
import copy

## [batch, seq_length, emb_size] ##
query = tf.Variable(tf.constant( \
    [[[1.1, 4.2, 1.3, 1.4], \
     [0.1, 2.3, 2.5, 2.7], \
     [3.1, 3.4, 3.8, 3.0]],\

    [[1.1, 1.2, 1.3, 1.4], \
     [2.1, 2.2, 2.3, 2.4], \
     [3.1, 3.2, 3.3, 3.4]],\

    [[0.1, 0.2, 0.3, 0.4], \
     [1.1, 1.2, 2.3, 2.4], \
     [0.1, 0.2, 3.3, 3.4]]], \
    dtype=tf.float32), name='query')
key   = tf.Variable(tf.constant( \
    [[[1.1, 4.2, 1.3, 1.4], \
     [0.1, 2.3, 2.5, 2.7], \
     [3.1, 3.4, 3.8, 3.0]],\

    [[1.1, 1.2, 1.3, 1.4], \
     [2.1, 2.2, 2.3, 2.4], \
     [3.1, 3.2, 3.3, 3.4]],\

    [[0.1, 0.2, 0.3, 0.4], \
     [1.1, 1.2, 2.3, 2.4], \
     [0.1, 0.2, 3.3, 3.4]]], \
    dtype=tf.float32), name='key')
value = tf.Variable(tf.constant( \
    [[[1.1, 4.2, 1.3, 1.4], \
     [0.1, 2.3, 2.5, 2.7], \
     [3.1, 3.4, 3.8, 3.0]],\

    [[1.1, 1.2, 1.3, 1.4], \
     [2.1, 2.2, 2.3, 2.4], \
     [3.1, 3.2, 3.3, 3.4]],\

    [[0.1, 0.2, 0.3, 0.4], \
     [1.1, 1.2, 2.3, 2.4], \
     [0.1, 0.2, 3.3, 3.4]]], \
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
  z = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
  ##here equal as, tf.matmul(query, key, transpose_b=True)
  print sess.run(z)

  scaled_z = tf.multiply(z, 1/tf.sqrt(4.0))
  print '='*20, 'query*key*scaled'

  print sess.run(scaled_z)
  print '='*20, 'softmax(query*key*scaled)'
  softmax_z= tf.nn.softmax(scaled_z, dim=2)
  ##here equal as, tf.nn.softmax(scaled_z, dim=-1)
  ##here equal as, tf.nn.softmax(scaled_z)
  print sess.run(softmax_z)

  print '='*20, 'query*key x value', '='*10, ' method-1'
  print sess.run(tf.matmul(softmax_z, value))
