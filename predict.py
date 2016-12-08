import os
import sys
import time
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data

n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

def rnn_model(x, weights, biases):
	"""Build a rnn model for image"""
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, n_input])
	x = tf.split(0, n_steps, x)

	lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
	outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights) + biases

def predict():
	"""Predict unseen images"""
	"""Step 0: load data and trained model"""
	mnist = input_data.read_data_sets("./data/", one_hot=True)
	checkpoint_dir = sys.argv[1]

	"""Step 1: build the rnn model"""
	x = tf.placeholder("float", [None, n_steps, n_input])
	y = tf.placeholder("float", [None, n_classes])

	weights = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weights')
	biases = tf.Variable(tf.random_normal([n_classes]), name='biases')

	pred = rnn_model(x, weights, biases)
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	"""Step 2: predict new images with the trained model"""
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		"""Step 2.0: load the trained model"""
		checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
		print('Loaded the trained model: {}'.format(checkpoint_file))

		saver = tf.train.Saver()
		saver.restore(sess, checkpoint_file)

		"""Step 2.1: predict new data"""
		test_len = 500
		test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
		test_label = mnist.test.labels[:test_len]
		print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

if __name__ == '__main__':
	predict()
