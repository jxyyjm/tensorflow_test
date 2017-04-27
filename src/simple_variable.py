#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import print_function

import tensorflow as tf

## 怎么共享变量呢？？##
## 变量作用域 ##
## 1.  tg.get_variable(name, shape, initializer) ## 通过所给的名字创建或是返回一个已经存在于当前域内的变量
## 2.  tf.variable_scope(scope_name, reuse=True/False) ## 指定当前的  命名空间  ##
## 3.  scope.reuse_variables() ## or ## tf.variable_scope(scope, reuse) ## 设定共享 ##
## 4.  current_scope = tg.get_variable_scope()  ## 获取当前的scope ##


## 1) creat a new variable ##
'''
with tf.variable_scope('foo'):
	with tf.variable_scope('bar'):
		v = tf.get_variable('v', [1])
		assert v.name == 'foo/bar/v:0'
		assert v.name == 'foo/v:0'
'''


## 2) 在其他地方定义变量时，共享之前定义过的变量 ##
'''
with tf.variable_scope('foo'):
	v = tf.get_variable('v', [1])
with tf.variable_scope('foo', reuse=True): ## 如果这里False，重复命名是error ##
	v1 = tf.get_variable('v', [1])
assert v1 == v
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
print ( 'v  is    : ', sess.run(v),  v.name)
print ( 'v1 is    : ', sess.run(v1), v.name )
'''


## 3) 在域内定义变量时，直接设置resuse共享  ##
'''
with tf.variable_scope('foo') as scope_now:
	v = tf.get_variable('v', [1])
	scope_now.reuse_variables() ######## scope_name.reuse_variables() #########
	v1 = tf.get_variable('v', [1]) 
	## 如果shape不一致，是error ## 之前不存在的变量的共享，也是error ##
	sess = tf.Session()
	init = tf.initialize_all_variables()
	sess.run(init)
	print ( 'v  is    : ', sess.run(v),  v.name)
	print ( 'v1 is    : ', sess.run(v1), v.name )
'''
	
## 4) 获取当前的变量作用域
'''
with tf.variable_scope('foo'):
	scope_here = tf.get_variable_scope()
	print ('current variable scope is : ', scope_here)
with tf.variable_scope('kkk'):
	scope_here = tf.get_variable_scope()
	print ('current variable scope is : ', scope_here)
'''
## 5) 域之间的切换完全 独立 ## 互不影响 ##
'''
with tf.variable_scope('foo') as foo_scope:
	print ('scope name is :  ', foo_scope.name)
with tf.variable_scope('bar'):
	with tf.variable_scope('baz') as other_scope:
		print ('other scope name is :  ', other_scope.name)
		with tf.variable_scope(foo_scope) as foo_scope2: ## 这里域与域的切换是完全独立的 #
			print ('foo_scope2 name is :   ', foo_scope2.name)
'''
## 6) 作用域的初始化器，对所有域内变量的创建 提供初始化函数 ##
'''
with tf.variable_scope('foo', initializer = tf.constant_initializer(0.4)):
	v = tf.get_variable('v', [1])
	w = tf.get_variable('w', [1], initializer=tf.constant_initializer(0.3)) ## 单独修改 #
	k = tf.get_variable('k', [1])
	sess = tf.Session()
	init = tf.initialize_all_variables()
	sess.run(init)
	current_scope = tf.get_variable_scope()
	print ('v is : ', sess.run(v), ';\nscope name is : ' ,current_scope.name, ' ;\nscope here is : ', current_scope)
	####
	print ('w is : ', sess.run(w))
	print ('k is : ', sess.run(k))
'''
## 7) 稍微复杂的初始化函数调用 	
## 与tf.randoom_normal()很相似，只不过嵌入到参数选项中 ##
## https://www.tensorflow.org/api_docs/python/state_ops/sharing_variables
'''
init = tf.random_normal_initializer(mean=1.0, stddev=2.0, dtype=tf.float32)
tf.reset_default_graph() ###
with tf.Session(): ## 这样类似tf.InteractiveSession() ##
	cur_scope      = tf.get_variable_scope()
	print ('current scope name is : ', cur_scope.name, ' ##')
	v_in_cur_scope =  tf.get_variable('x', shape=[2, 4], initializer=init)
	v_in_cur_scope.initializer.run() ## 只有一个variable,用v.initializer.run()就够了 ##
	print (v_in_cur_scope.eval())
'''
	
## 8) names of ops in tf.variable_scope() ## ??? what's the meaning ??
#with tf.variable_scope("foo", reuse=True):
#	x = tf.get_variable('x', [1])
#	print ('x is ', x.name)
#	v = tf.get_variable('x') + 1.0
#	print ('v.op.name', v.op.name)
##  比较 ## 上面的，是因为直接用reuse=True; 但是后面紧跟着x的定义,默认为对x的定义，也使用reuse，但是是没有的，所以报错 ##
with tf.variable_scope("foo") as scope_cur:
	x = tf.get_variable('x', [1])
	print ('x is ', x.name)
	scope_cur.reuse_variables()
	v = tf.get_variable('x') + 1.0
	print ('v.name', v.name)
	print ('v.op.name', v.op.name)



