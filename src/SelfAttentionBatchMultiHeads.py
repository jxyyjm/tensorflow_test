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
## input_tensor    = [batch, seq_length, emb_size]
## input_matrix    = [batch*seq_length,  emb_size]
## query_layer     = input_matrix X dense_matrx = [batch*seq_length, hidden_size]
## key_layer       = input_matrix X dense_matrx = [batch*seq_length, hidden_size]
## value_layer     = input_matrix X dense_matrx = [batch*seq_length, hidden_size]

## query/key/value = [batch, seq_length, per_head_size, heads_num] ##
query = tf.Variable(tf.constant( \
    [[[1.1, 4.2, 1.3, 1.4, 1.1, 4.2, 1.3, 1.4], \
     [0.1, 2.3, 2.5, 2.7, 0.1, 2.3, 2.5, 2.7], \
     [3.1, 3.4, 3.8, 3.0, 3.1, 3.4, 3.8, 3.0]],\

    [[1.1, 1.2, 1.3, 1.4, 1.1, 1.2, 1.3, 1.4], \
     [2.1, 2.2, 2.3, 2.4, 2.1, 2.2, 2.3, 2.4], \
     [3.1, 3.2, 3.3, 3.4, 3.1, 3.2, 3.3, 3.4]],\

    [[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4], \
     [1.1, 1.2, 2.3, 2.4, 1.1, 1.2, 2.3, 2.4], \
     [0.1, 0.2, 3.3, 3.4, 0.1, 0.2, 3.3, 3.4]]], \
    dtype=tf.float32), name='query')

batch, seq_length, emb_size = query.get_shape().as_list()
## 重要 Dense层 是对emb_size到hidden_size的维度映射 ##
hidden_size = emb_size
heads_num   = 2 
size_per_head = int(hidden_size/heads_num)
## 1) 将多头的参数，也放入这个dense层的列里 
##    本来emb_size --> 一个 hidden_size'=size_per_head空间，现在复制了heads_num份 ##
##    hidden_size = size_per_head * heads_num ##
## 2) 由于是对emb_size向量空间的变化，将batch,seq_length 合并，这样每行代表一个token的在emb_size的表示vec
##    这种处理，有利于使用tf.matmul作向量空间的转换计算，理解方便 ## 
##    其实也可以 不作二维化, 用 tf.tensordot()来实现 ##
#query_2d    = tf.reshape(query, [batch*seq_length, emb_size])
#query_layer = tf.layers.dense(query_2d, units=hidden_size, use_bias=False, kernel_initializer=tf.ones_initializer())
## 假装经过 Dense 层
query_layer = tf.reshape(query, [batch*seq_length, hidden_size])
## 拆分成batch样式 ##
query_layer = tf.reshape(query_layer, [batch, seq_length, heads_num, size_per_head])
## 转置出heads_num到dim=1 ## 为后面用tf.matmul 计算attention 提供便利，它只计算后两维 ##
query_layer = tf.transpose(query_layer, [0, 2, 1, 3])

key   = tf.Variable(tf.constant( \
    [[[1.1, 4.2, 1.3, 1.4, 1.1, 4.2, 1.3, 1.4], \
     [0.1, 2.3, 2.5, 2.7, 0.1, 2.3, 2.5, 2.7], \
     [3.1, 3.4, 3.8, 3.0, 3.1, 3.4, 3.8, 3.0]],\

    [[1.1, 1.2, 1.3, 1.4, 1.1, 1.2, 1.3, 1.4], \
     [2.1, 2.2, 2.3, 2.4, 2.1, 2.2, 2.3, 2.4], \
     [3.1, 3.2, 3.3, 3.4, 3.1, 3.2, 3.3, 3.4]],\

    [[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4], \
     [1.1, 1.2, 2.3, 2.4, 1.1, 1.2, 2.3, 2.4], \
     [0.1, 0.2, 3.3, 3.4, 0.1, 0.2, 3.3, 3.4]]], \
    dtype=tf.float32), name='key')
## 重要： Dense层 是对emb_size到hidden_size的维度映射 ##
#key_2d    = tf.reshape(key, [batch*seq_length, emb_size])
#key_layer = tf.layers.dense(key_2d, units=hidden_size, use_bias=False, kernel_initializer=tf.ones_initializer())
## 假装经过 Dense 层 ##
key_layer = tf.reshape(key, [batch*seq_length, hidden_size])
key_layer = tf.reshape(key_layer, [batch, seq_length, heads_num, size_per_head])
key_layer = tf.transpose(key_layer, [0, 2, 1, 3])
value = tf.Variable(tf.constant( \
    [[[1.1, 4.2, 1.3, 1.4, 1.1, 4.2, 1.3, 1.4], \
     [0.1, 2.3, 2.5, 2.7, 0.1, 2.3, 2.5, 2.7], \
     [3.1, 3.4, 3.8, 3.0, 3.1, 3.4, 3.8, 3.0]],\

    [[1.1, 1.2, 1.3, 1.4, 1.1, 1.2, 1.3, 1.4], \
     [2.1, 2.2, 2.3, 2.4, 2.1, 2.2, 2.3, 2.4], \
     [3.1, 3.2, 3.3, 3.4, 3.1, 3.2, 3.3, 3.4]],\

    [[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4], \
     [1.1, 1.2, 2.3, 2.4, 1.1, 1.2, 2.3, 2.4], \
     [0.1, 0.2, 3.3, 3.4, 0.1, 0.2, 3.3, 3.4]]], \
    dtype=tf.float32), name='value')
## 重要:  Dense层 是对emb_size到hidden_size的维度映射 ##
#value_2d    = tf.reshape(value, [batch*seq_length, emb_size])
#value_layer = tf.layers.dense(value_2d, units=hidden_size, use_bias=False, kernel_initializer=tf.ones_initializer())
## 假装经过 Dense 层 ##
value_layer = tf.reshape(value, [batch*seq_length, hidden_size])
value_layer = tf.reshape(value_layer, [batch, seq_length, heads_num, size_per_head])
value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

gpu_options = tf.GPUOptions(allow_growth = True)
with tf.Session(config = tf.ConfigProto( \
                gpu_options = gpu_options, \
                allow_soft_placement = True, \
                log_device_placement = False)) as sess:
  sess.run(tf.global_variables_initializer())
  print '='*20, 'query'
  print sess.run(query)
  print '='*20, 'query_layer'
  print sess.run(query_layer)
  print '='*20, 'key'
  print sess.run(key)
  print '='*20, 'key_layer'
  print sess.run(key_layer)
  print '='*20, 'value'
  print sess.run(value)
  print '='*20, 'value_layer'
  print sess.run(value_layer)
  print '='*20, 'query*key'
  z = tf.matmul(query_layer, tf.transpose(key_layer, [0, 1, 3, 2]))
  ##here equal as, tf.matmul(query_layer, key_layer, transpose_b=True)
  print sess.run(z)
  
  scaled_z = tf.multiply(z, 1/tf.sqrt(4.0))
  print '='*20, 'query*key*scaled'

  print sess.run(scaled_z)
  print '='*20, 'softmax(query*key*scaled)'
  softmax_z= tf.nn.softmax(scaled_z, dim=3) ## 需要指定沿着哪个维 作softmax ##
  ##here equal as, tf.nn.softmax(scaled_z, dim=-1) ## 建议使用这个，默认也是沿着最后一维作softmax ##
  ##here equal as, tf.nn.softmax(scaled_z)
  print sess.run(softmax_z)

  print '='*20, 'query*key x value', '='*10, ' method-1'
  print sess.run(tf.matmul(softmax_z, value_layer))


