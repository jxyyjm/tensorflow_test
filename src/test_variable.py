#!/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf
## varibale create / initilize / save / reload ##
## 变量存在于内存中，其中保存着张量 ##
## tf.Variable class && tf.train.Saver class
## 1)  creat variable ##
## 向类传递 张量 以创建变量 ##
## 1.1)  tensors 种类 及 函数 ## dtype, seed ##
##		 1.1.1 constant value tensor
##				tf.zeros(shape, dtype=tf.float32, name=None);
##				tf.zeros_like(tensor, dtype=None, name=None, optimize=True);
##				tf.ones();
##				tf.ones_like();
##				tf.fill(dims, value, name=None); e.g. tf.fill([2,3], -.1) ##
##				tf.constant(value, shape=None); e.g. tf.constant(-1.0, shape=[2, 3]) ## 
##		 1.1.2 sequences             ## 队列的 tensor ##
##				tf.linspace(start, stop, num, name=None); 浮点型
##				tf.range(start, limit=None, delta=1);     整数型
##		 1.1.3 random tensor         ## 按照不同分布 ##
##				tf.random_normal(shape, mean=0.0, stddev=1.0)    ## 正态分布 ##
##				tf.truncated_normal(shape, mean=0.0, stddev=1.0) ## 截断正态分布 ##
##				tf.random_uniform(shape, minval=0, maxval=None)  ## 均匀分布 ##
##				tf.random_shuffle(value, seed, name)             ## 对行shuffle  ##
##				tf.random_crop(value, size)  ## 随机裁剪 ##
##				tf.multinomial(logits, num_samples)              ## ??? ##
##				tf.random_gamma(shape, alpha, beta=None)         ## 伽马分布 ##
##				tf.set_random_seed(seed)     ##  设置所有分布的seed，随机产生可重复 ##
## 1.2) create a tensor which will initialize a variable
##		init_tensor = tf.random_uniform([10, 5], minval=0, maxval=10, dtype=tf.float32)
##		weight      = tf.Variable(init_tensor)
##		sess        = tf.Session()
##		init        = tf.initialize_all_variables()
##		sess.run(init); sess.run(weight); sess.run(init_tensor); sess.close()
## 1.3) initialize a variable with another variable
##		用一个变量去初始化另一个变量，必须使得自己 初始化完成Variable.initialized_value
##		weight      = tf.Variable(tf.zeros([2,3], dtype=tf.float32)
##		weight2     = tf.Variable(weight.initialized_value()) ## 和下面的函数不一样 ##
##		init        = tf.initialize_all_variables() ##tf.global_variables_initializer()#
##		sess = tf.Session(); sess.run(init); sess.run(weight2); sess.run(weight)
## 1.4) save && restore variables 
##		variable1   = tf.Variable(tf.zeros([2,2], dtype=tf.int32), name='variable1')
##		variable2   = tf.Variable(tf.ones_like(variable1.initialized_value()), name='variable2')
##		init        = tf.initialize_all_variables()
##		sess        = tf.Session()
##		sess.run(init); sess.run(variable1); sess.run(variable2)
##		saver       = tf.train.Saver()
##		save_path  = saver.save(sess, './log'); sess.close()  ## 保存变量 ##
##		## restore ##
##		sess2       = tf.Session()
##		saver.restore(sess2, './log')         ### 重新加载，直接可以使用，不用初始化 ##
##		sess2.run(variable1); sess2.run(variable2)
##		####################################################################
##		指定 哪些variable保存 ##
##		saver = tf.train.Saver({'mode1_variable1':variable1, 'mode2_variable2':variable2})
##		## 这里的指定的名称，是Variable等号前面的那个名字 ## 


