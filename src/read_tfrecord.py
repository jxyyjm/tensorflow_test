#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
## tf 1.4.1 ##

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

## method 1: iterator read directly ##
record_iterator = tf.python_io.tf_record_iterator(path='./a')
## An iterator that read the records from a TFRecords file ##
for i, string_record in enumerate(record_iterator):
  example = tf.train.Example()
  example.ParseFromString(string_record)
  print (i+1, 'example', '*'*20)
  print (example.features)
  print (i+1, 'example', 'x0.value', '*'*20)
  print (example.features.feature['x0'])

  print (i+1, 'example', 'x1.value', '*'*20)
  print (example.features.feature['x1'])

  print (i+1, 'example', 'x2.value', '*'*20)
  print (example.features.feature['x2'])
print ('method Iterator done', '='*20)  

## method 2: queue + single_parse ##
## parse feature ##
features = { 
  'x0': tf.FixedLenFeature([4], tf.int64), 
  'x1': tf.FixedLenFeature([5], tf.float32),
  'x2': tf.FixedLenFeature([3], tf.string)
}
## notice: [num] must be give ##
file_queue = tf.train.string_input_producer(['./a'])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(file_queue)
example= tf.parse_single_example(serialized_example, features=features)

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options = tf.GPUOptions(allow_growth=True))
with tf.Session(config = config) as sess:
  sess.run(tf.global_variables_initializer())
  tf.train.start_queue_runners(coord=tf.train.Coordinator())
  for i in range(5):
    x = sess.run(example)
    print ('iter:\t', i, '*'*10)
    print ('x.shape', len(x), x['x2'], 'x[x2].shape', x['x2'].shape, x['x2'][2].decode('utf-8'))
print ('method queue + single_parse done', '='*20, '\n')

## method 3: dataset ##
def decode_line(value):
  features = {
    'x0': tf.FixedLenFeature([4], tf.int64),
    'x1': tf.FixedLenFeature([5], tf.float32),
    'x2': tf.FixedLenFeature([3], tf.string)
  }
  return tf.parse_single_example(value, features=features)

dataset = tf.data.TFRecordDataset(['./a'])
dataset = dataset.map(decode_line)
dataset = dataset.batch(batch_size=3)
dataset = dataset.repeat(count=2)

iterator = dataset.make_one_shot_iterator()
next_example = iterator.get_next()

config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True))
with tf.Session(config = config) as sess:
  sess.run(tf.global_variables_initializer())
  num  = 0
  while True:
    try:
      x = sess.run(next_example)
    except tf.errors.OutOfRangeError:
      break
    print ('iter:\t', num, '*'*10)
    print ('x.shape', len(x), x['x2'], 'x[x2].shape', x['x2'].shape, x['x2'][0,2].decode('utf-8'))
    num += 1
print ('method queue + single_parse done', '='*20, '\n')
