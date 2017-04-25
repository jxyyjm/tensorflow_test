#!/usr/bin/python
# -*- coding:utf-8 -*-
# tensorFlow 让我们描述一个交互操作图，然后完全将其运行在Python外部 #

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
# 构造一个占位符矩阵, 大小[n, 784] ## 前面的n可以为任意大小 ##
# Inserts a placeholder for a tensor that will be always fed.
# **Important**: This tensor will produce an error if evaluated. 
# Its value must be fed using the `feed_dict` optional argument to `Session.run()`,    `Tensor.eval()`, or `Operation.run()`.

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 构造两个功能等同于张量的variable, 在图中可被修改，在计算中可被修改 ##

## 构造目标模型 ##
y = tf.nn.softmax(tf.matmul(x, W) + b)

## 构造代价函数 ##
y_ = tf.placeholder('float', [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  ## ? 这里是什么意思？交叉熵 ##
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

## 优化方法设定 ##
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

## 初始化 ##
#init = tf.global_variables_initializer() 这个地方好像是有问题的 ##
init = tf.initialize_all_variables()
sess = tf.Session() ## 构造一个对话 ##
sess.run(init)

## 训练 ##
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

## 评估 ##
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
#print accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
