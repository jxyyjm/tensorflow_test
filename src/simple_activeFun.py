#!/usr/bin/python
# -*- coding:utf-8 -*-

# @time      : 2017-02-17
# @author    : yujianmin
# @reference : 
# @what to-do: active-function in tf.nn 

from __future__ import print_function
import tensorflow as tf

## 1) 激活函数 ## activation functions
## 1.1 tf.nn.relu(features) ; rectified linear units; output = max(feat_values, 0)
##		reference: http://www.cnblogs.com/neopenx/p/4453161.html
##		reference: Rectified Linear Units Improve Restricted Boltzmann Machines
## 1.2 tf.nn.relu6(features);  output = min( max(features, 0), 6 ) 先取max(feat,0) 再与6比较取小 ## 限制上限为 6
## 1.3 tf.nn.crelu(features); 
##		reference: Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units
## 1.4 tf.nn.elu(features)  ;  output = exp(feature) -1 if features<0, else feature;
##		reference: http://arxiv.org/abs/1511.07289
##		reference: Fast and Accurate Deep Network Learning by Exponential Linear Units
## 1.5 tf.nn.softplus(features); output = log( exp(features)+1 )
## 1.6 tf.nn.softsign(features); output = features / ( abs(features)+1 )

x = tf.Variable(tf.random_normal([1,10], mean=0.0, stddev=1.0))
sess = tf.Session()
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)
print ('x          : ', sess.run(x))
y = tf.nn.relu(x)
print ('tf.nn.relu : ', sess.run(y))
y = tf.nn.relu6(x)
print ('tf.nn.relu6: ', sess.run(y))
##y = tf.nn.crelu(x) ## this version has no this function ##
y = tf.nn.elu(x)
print ('tf.nn.elu  : ', sess.run(y))
y = tf.nn.softplus(x)
print ('tf.nn.softplus',sess.run(y))
y = tf.nn.softsign(x)
print ('tf.nn.softsign:',sess.run(y))

## 2) trik
## 2.1 tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
##		为了克服overfitting, 随机选择哪些连接暂时中断。有点类似于spare稀疏 ##
##		With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, otherwise outputs 0.
##		reference: Dropout: A Simple Way to Prevent Neural Networks from Overfitting
## 3) tf.nn.bias_add(value, bias) ## 将 1-D的bias加到 tensor-value上
## 4) tf.nn.sigmoid(x, name);  output = 1 /(1+exp(-x))
## 5) tf.nn.tanh(x, name);    
##		tanh(x) = sinh(x)/cosh(x) 
##				= [e^(x)-e^(-x)]/[e^(x)+e^(-x)]
##				= [e^(2x)-1]/[e^(2x)+1]



tf.reset_default_graph() ## tf.Session()是建立在default-graph上的，为了避免default-graph 存在一些dead nodes，影响到当前 session. 	
with tf.Session() as sess:
	## 定义一个共享变量，初始化自身，并初始化其他变量 ##
	with tf.variable_scope('foo') as scope_name:
		x = tf.get_variable('x', [1,7], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
		# wrong, if use x = tf.Variable() # not share-variable-type-functioin #
		# 在命名空间内，需要用tf.get_variable来定义，不能用tf.Variable ##
		sess.run(x.initializer)
		#x.initializer.run(session=sess) or as above initialize ##
		print ('x.name : ', x.name)
		scope_name.reuse_variables()
		print ('scope current is : ', tf.get_variable_scope().name)
		x_ = tf.get_variable('x')
		print ('in this scope, x  is : ', sess.run(x))
		print ('in this scope, x_ is : ', sess.run(x_))
	print ('###########################')
	print ('###  has not in above scope, the variable；x & x_ still canbe used ###')
	print ('scope current is : ', tf.get_variable_scope().name)
	print ('x is : ', sess.run(x))
	print ('x_ is : ', sess.run(x_))
	print ('##########################')
	y = tf.nn.dropout(x_, keep_prob=0.5)
	print ('dropout(x_) is :', sess.run(y))
	y = tf.nn.sigmoid(x_)
	print ('sigmoid     is :', sess.run(y))
	y = tf.nn.tanh(x_)
	print ('tanh        is :', sess.run(y))

### notice :: scope 是为了解决变量共享，借助之前的变量来生成其他变量 ##
### notice :: variable 的生命周期取决于会话的生命。所以scope内的变量其他地方也是可以访问到的 ##
### notice :: session 是为了与TensorFlow系统进行交互的 ##











