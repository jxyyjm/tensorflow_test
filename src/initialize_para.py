#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
	use saved para-value to initialize new-variable
'''
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tensorflow as tf
from dataset_base import get_dataset

def create_varibale(init_value, name):
	return tf.Variable( init_value, dtype=tf.float32, name=name)

dataset_ = get_dataset('./iris.data', batch_size=10, buffer_size=100, epoch=10)

dir(tf.contrib)
with tf.Session() as sess:
	saver = tf.train.import_meta_graph('./tmp/my-model.ckpt-4951.meta')
	saver.restore(sess,tf.train.latest_checkpoint('./tmp/'))

	graph = tf.get_default_graph()
	print ('first after restore graph')
	print [i.name for i in tf.trainable_variables()]

	## 1) variable-value in reloaded graph ##
	w1_init = sess.run(graph.get_tensor_by_name('layer-1/w1:0'))
	w2_init = sess.run(graph.get_tensor_by_name('layer-2/w2:0'))
	b1_init = sess.run(graph.get_tensor_by_name('layer-1/b1:0'))
	b2_init = sess.run(graph.get_tensor_by_name('layer-2/b2:0'))
	## 2) placeholder in reloaded graph ##
	x  = graph.get_operation_by_name('compute/x').outputs[0] ## notice: placeholder is an operation ## 
	y  = graph.get_operation_by_name('compute/y').outputs[0]
	print ('get placeholder from graph')
	print ('x.outputs[0]', x)
	## create new
	w1 = create_varibale(w1_init, 'w1')
	w2 = create_varibale(w2_init, 'w2')
	b1 = create_varibale(b1_init, 'b1')
	b2 = create_varibale(b2_init, 'b2')
	print ('after create new variable')
	print [i.name for i in tf.trainable_variables()]
	
	## 3) operation in reloaded graph ##
	pred = tf.matmul(tf.matmul(x, w1)+b1, w2)+b2
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= pred))
	accuracy = tf.contrib.metrics.accuracy(\
					labels= tf.argmax(y, 1), \
					predictions = tf.argmax(tf.nn.softmax(pred, 1), 1))
	train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
	cur_saver = tf.train.Saver([w1, b1, w2, b2], max_to_keep=2)
	sess.run(tf.global_variables_initializer())
	iter = 0
	while True:
		try:
			input, label = sess.run(dataset_.get_next())
			label        = sess.run(tf.squeeze(tf.one_hot(label, 3), 1))
		except tf.errors.OutOfRangeError:
			print 'next is end'
			break
		iter += 1
		loss_, accu_ = sess.run([loss, accuracy], feed_dict={x:input, y:label})
		sess.run(train_op, feed_dict={x:input, y:label})
		print 'iter:\t', iter, '\tloss_:\t', loss_, '\taccu_:\t', accu_
		if iter % 50 == 1:
			save_path = cur_saver.save(sess, './tmp2/my-model.ckpt', global_step=iter)
	print ('model_save_path:', save_path)















