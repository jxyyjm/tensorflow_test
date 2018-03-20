#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf

## method - 1 import para#
#with tf.variable_scope('layer-1'):
#	w1 = tf.Variable( \
#			tf.random_normal(shape = [4,8], mean  = 0.0, stddev= 1.0),\
#			dtype = tf.float32, \
#			name  = 'w1')
#	b1 = tf.Variable(tf.zeros([1,8]), dtype = tf.float32, name = 'b1')
#with tf.variable_scope('layer-2'):
#	w2 = tf.Variable( \
#			tf.random_normal(shape = [8,3], mean  = 0.0, stddev= 1.0),\
#			dtype = tf.float32, \
#			name  = 'w2')
#	b2 = tf.Variable(tf.zeros([1,3]), dtype = tf.float32, name = 'b2')
#
#saver = tf.train.Saver()
#with tf.Session() as sess:
#	model_file=tf.train.latest_checkpoint('./tmp/')
#	saver.restore(sess, model_file)
#	print (sess.run(w1))
#	print (type(sess.run(w1)))

	#help(saver)


# method - 2 import para#
#saver = tf.train.load_checkpoint('./tmp') ## 可以直接读取最新的checkpoint ##
#a = saver.get_variable_to_shape_map()
#for i in a:
#	print i,':', saver.get_tensor(i).shape


## method - 3 import para#
#from tensorflow.python import pywrap_tensorflow
#checkpoint_path = './tmp/my-model.ckpt-1451'
#reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
#var_to_shape_map = reader.get_variable_to_shape_map()
#for key in var_to_shape_map:
#	print(key, reader.get_tensor(key))



## method - 1 import Graph and variable from 0##
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


next_data = get_batch_data('./iris.data', batch_size=30, epoch=10)

dir(tf.contrib)
## if has no this code, will error: No op named OneShotIterator in defined operations. #
saver = tf.train.import_meta_graph('./tmp/my-model.ckpt-101.meta')
with tf.Session() as sess:
	saver.restore(sess,tf.train.latest_checkpoint('./tmp/'))
	value = sess.run('layer-1/w1:0')
	w_aga = tf.Variable( \
			initial_value = tf.constant( \
				value, dtype=tf.float32), \
			name = 'w_aga')
	print ('before init, w_aga', w_aga)
	sess.run(w_aga.initializer)
	print ('after  init, w_aga', w_aga)
	print ('after  init, w_aga.run', sess.run(w_aga))

	graph = tf.get_default_graph()
	print (sess.run(graph.get_tensor_by_name('layer-1/w1:0'))) ## 具体值 ##
	variable_trainable = [c.name for c in tf.trainable_variables()] ## 图中的 所有可训练变量 ##
	#variable_all = [c.name for c in tf.global_variables()] ## 图中的 所有变量 ##
	#train_op = tf.get_collection('train_op')[0]
	loss     = tf.get_collection('loss')[0]
	accuracy = tf.get_collection('accuracy')[0]
	train_op = tf.get_collection('train_op')[0] ## 获取保存的 图操作 ##
	print ('train_op', train_op)
	print ('graph', graph.get_all_collection_keys())
	x = graph.get_operation_by_name('x').outputs[0] ## 获取placeholder 变量 ##
	y = graph.get_operation_by_name('y').outputs[0]
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	try:
		iter = 0
		while True:
			input, label = sess.run(next_data)
			#test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict={x:input, y:label})
			#print iter, '\ttest-loss', test_loss, '\ttest-accuracy', test_accuracy
			sess.run(train_op, feed_dict={x:input, y:label})
			if iter % 50 == 1:
				test_loss, test_accu = sess.run([loss, accuracy], feed_dict={x:input, y:label})
				print iter, '\ttest_loss:', test_loss, '\ttest-accuracy:', test_accu
			iter += 1
	except tf.errors.OutOfRangeError:
		print 'end'





