#!/usr/bin/python
# -*- coding:utf-8 -*-

# @time      : 2017-04-27
# @author    : yujianmin
# @reference :
# @what to-do: get some-knowledge of variable in tensorflow

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
##		init        = tf.global_variables_initializer()
##		sess.run(init); sess.run(weight); sess.run(init_tensor); sess.close()
## 1.3) initialize a variable with another variable
##		用一个变量去初始化另一个变量，必须使得自己 初始化完成Variable.initialized_value
##		weight      = tf.Variable(tf.zeros([2,3], dtype=tf.float32)
##		weight2     = tf.Variable(weight.initialized_value()) ## 和下面的函数不一样 ##
##		init        = tf.global_variables_initializer() ##tf.global_variables_initializer()#
##		sess = tf.Session(); sess.run(init); sess.run(weight2); sess.run(weight)
## 1.4) save && restore variables 
##		variable1   = tf.Variable(tf.zeros([2,2], dtype=tf.int32), name='variable1')
##		variable2   = tf.Variable(tf.ones_like(variable1.initialized_value()), name='variable2')
##		init        = tf.global_variables_initializer()
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

from __future__ import print_function
import tensorflow as tf

print ('## 怎么共享变量呢？？##')
print ('## 变量作用域 ##')
print ('## 1.  tg.get_variable(name, shape, initializer) ## 通过所给的名字创建或是返回一个已经存在于当前域内的变量')
print ('## 2.  tf.variable_scope(scope_name, reuse=True/False) ## 指定当前的  命名空间  ##')
print ('## 3.  scope.reuse_variables() ## or ## tf.variable_scope(scope, reuse) ## 设定共享 ##')
print ('## 4.  current_scope = tg.get_variable_scope()  ## 获取当前的scope ##')


print ('\n## 1) creat a new variable ##')
with tf.variable_scope('foo'):
	with tf.variable_scope('bar'):
		v = tf.get_variable('v', [1])
		assert v.name == 'foo/bar/v:0'
		#assert v.name == 'foo/v:0'

tf.reset_default_graph()
print ('\n## 2) 在其他地方定义变量时，共享之前定义过的变量 ##')
with tf.variable_scope('foo'):
	v = tf.get_variable('v', [1])
with tf.variable_scope('foo', reuse=True): ## 如果这里False，重复命名是error ##
	v1 = tf.get_variable('v', [1])
assert v1 == v
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print ( 'v  is    : ', sess.run(v),  v.name)
print ( 'v1 is    : ', sess.run(v1), v.name )


tf.reset_default_graph()
print ('\n## 3) 在域内定义变量时，直接设置resuse共享  ##')
with tf.variable_scope('foo') as scope_now:
	v = tf.get_variable('v', [1])
	scope_now.reuse_variables() ######## scope_name.reuse_variables() #########
	v1 = tf.get_variable('v', [1]) 
	## 如果shape不一致，是error ## 之前不存在的变量的共享，也是error ##
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	print ( 'v  is    : ', sess.run(v),  v.name)
	print ( 'v1 is    : ', sess.run(v1), v.name )
	
tf.reset_default_graph()
print ('\n## 4) 获取当前的变量作用域 ##')
with tf.variable_scope('foo'):
	scope_here = tf.get_variable_scope()
	print ('current variable scope is : ', scope_here)
with tf.variable_scope('kkk'):
	scope_here = tf.get_variable_scope()
	print ('current variable scope is : ', scope_here)


tf.reset_default_graph()
print ('\n## 5) 域之间的切换完全 独立 ## 互不影响 ##')
with tf.variable_scope('foo') as foo_scope:
	print ('scope name is :  ', foo_scope.name)
with tf.variable_scope('bar'):
	with tf.variable_scope('baz') as other_scope:
		print ('other scope name is :  ', other_scope.name)
		with tf.variable_scope(foo_scope) as foo_scope2: ## 这里域与域的切换是完全独立的 #
			print ('foo_scope2 name is :   ', foo_scope2.name)

tf.reset_default_graph()
print ('\n## 6) 作用域的初始化器，对所有域内变量的创建 提供初始化函数 ##')
with tf.variable_scope('foo', initializer = tf.constant_initializer(0.4)):
	v = tf.get_variable('v', [1])
	w = tf.get_variable('w', [1], initializer=tf.constant_initializer(0.3)) ## 单独修改 #
	k = tf.get_variable('k', [1])
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	current_scope = tf.get_variable_scope()
	print ('v is : ', sess.run(v), ';\nscope name is : ' ,current_scope.name, ' ;\nscope here is : ', current_scope)
	####
	print ('w is : ', sess.run(w))
	print ('k is : ', sess.run(k))


tf.reset_default_graph()
print ('\n## 7) 稍微复杂的初始化函数调用 	##')
print ('## 与tf.randoom_normal()很相似，只不过嵌入到参数选项中 ##')
print ('## https://www.tensorflow.org/api_docs/python/state_ops/sharing_variables ##')
init = tf.random_normal_initializer(mean=1.0, stddev=2.0, dtype=tf.float32)
tf.reset_default_graph() ###
with tf.Session(): ## 这样类似tf.InteractiveSession() ##
	cur_scope      = tf.get_variable_scope()
	print ('current scope name is : ', cur_scope.name, ' ##')
	v_in_cur_scope =  tf.get_variable('x', shape=[2, 4], initializer=init)
	v_in_cur_scope.initializer.run() ## 只有一个variable,用v.initializer.run()就够了 ##
	print (v_in_cur_scope.eval())

	
tf.reset_default_graph()
print ('\n## 8) names of ops in tf.variable_scope() ## ??? what is the meaning ?? ##')
#with tf.variable_scope('foo', reuse=True): 
#	x = tf.get_variable('x', [1]) 
#	print ('x is ', x.name) 
#	v = tf.get_variable('x') + 1.0 
#	print ('v.op.name', v.op.name) 
##  比较 ## 上面的，是因为直接用reuse=True; 但是后面紧跟着x的定义,默认为对x的定义，也使用reuse，但是是没有的，所以报错 ##"
with tf.variable_scope("foo") as scope_cur:
	x = tf.get_variable('x', [1])
	print ('x is ', x.name)
	scope_cur.reuse_variables()
	v = tf.get_variable('x') + 1.0
	print ('v.name', v.name)
	print ('v.op.name', v.op.name)



