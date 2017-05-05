#!/usr/bin/python


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def sample_prob(probs):
	return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

mnist  = input_data.read_data_sets("MNIST_data/", one_hot=True)
trainX = mnist.train.images
trainY = mnist.train.labels
testX  = mnist.test.images
testY  = mnist.test.labels
print ('trainX')
print (trainX.shape)
print (trainX)

n_sample, n_feature = trainX.shape
n_hidden = 200

X = tf.placeholder(tf.float32, shape=[None, n_feature])
Y = tf.placeholder(tf.float32, shape=[None, 10])

rbm_w = tf.Variable(tf.zeros(shape=[n_feature, n_hidden], dtype=tf.float32))
rbm_vb= tf.Variable(tf.zeros(shape=[n_feature],dtype=tf.float32))
rbm_hb= tf.Variable(tf.zeros(shape=[n_hidden], dtype=tf.float32))

h0 = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb))
v1 = sample_prob(tf.nn.sigmoid(tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print ('h0')
tmp = sess.run(h0, feed_dict={X:trainX, Y:trainY})
print (tmp.shape)
print (tmp)
print ('v1')
tmp = sess.run(v1, feed_dict={X:trainX, Y:trainY})
print (tmp.shape)
print (tmp)
