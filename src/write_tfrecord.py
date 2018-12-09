# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
## tf 1.4.1 ##

## 生成 特征 ##
def create_int64_feature(values):
  """Returns an int64_list from a bool / enum / int / uint."""
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature
def create_float_feature(values):
  """Returns a float_list from a float / double."""
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature
def create_bytes_feature(values):
  """Returns a bytes_list from a string / byte."""
  feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))
  return feature

## 转换成 tf-record ## 
def trans2tfrecord(_input):
  example = tf.train.Example( ## example 协议 ##
    features = tf.train.Features( ## key-value 的字典 ##
      feature = { 
        'x0': create_int64_feature(_input[0]), 
        'x1': create_float_feature(_input[1]), 
        'x2': create_bytes_feature(_input[2])
        }   
      )   
    )
  return example
## 转换 tf-record 2 string ##
def trans2serialized(tfrecord):
  return tfrecord.SerializeToString()


## data-input ##
_input = [[], [], []]
_input[0] = [0, True, 1, False]
_input[1] = [0.2, 0.9, 1.4, 3e-2, 0.22222]
_input[2] = ['hello', 'word', '中国']

example = trans2tfrecord(_input)
serialized_example = trans2serialized(example)
## 写操作 ##
writer = tf.python_io.TFRecordWriter('./a')
writer.write(record=serialized_example)

_input[2] = ['hello', 'zz', 'mm']
example = trans2tfrecord(_input)
serialized_example = trans2serialized(example)
writer.write(record=serialized_example)

import random
for i in range(4):
  _input[0] = random.sample(range(100), 4)
  _input[1] = [random.normalvariate(0, 0.01*i) for i in range(5)]
  example = trans2tfrecord(_input)
  serialized_example = trans2serialized(example)
  writer.write(record=serialized_example)

writer.close()
