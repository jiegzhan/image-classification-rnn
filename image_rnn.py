import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

class ImageRNN(object):
	def __init__(self, n_input, n_steps, n_hidden, n_classes):
		self.input_x = tf.placeholder('float', [None, n_steps, n_input])
		self.input_y = tf.placeholder('float', [None, n_classes])

		self.input_x = tf.transpose(self.input_x, [1, 0, 2])
		self.input_x = tf.reshape(self.input_x, [-1, n_input])
		self.input_x = tf.split(0, n_steps, self.input_x)

		lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

		with tf.name_scope('output'):
			outputs, states = rnn.rnn(lstm_cell, self.input_x, dtype=tf.float32)
			weights = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weights')
			biases = tf.Variable(tf.random_normal([n_classes]), name='biases')
			self.prediction = tf.matmul(outputs[-1], weights) + biases

		with tf.name_scope('cost'):
			losses = tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.input_y)
			self.cost = tf.reduce_mean(losses)

		"""
		with tf.name_scope('num_correct'):
			self.num_correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.input_y, 1))

		with tf.name_scope('accuracy'):
			self.accuracy = tf.reduce_mean(tf.cast(self.num_correct, tf.float32))
		"""
