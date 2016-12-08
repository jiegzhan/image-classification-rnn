import os
import sys
import json
import time
import tensorflow as tf
from image_rnn import ImageRNN
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data

n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

def rnn_model(x, weights, biases):
	"""RNN (LSTM or GRU) model for image"""
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, n_input])
	x = tf.split(0, n_steps, x)

	lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
	outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights) + biases

def train():
	"""Step 0: load image data"""
	mnist = input_data.read_data_sets("./data/", one_hot=True)

	"""Step 1: load training parameters"""
	parameter_file = sys.argv[1]
	params = json.loads(open(parameter_file).read())

	"""
	x = tf.placeholder("float", [None, n_steps, n_input])
	y = tf.placeholder("float", [None, n_classes])

	weights = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weights')
	biases = tf.Variable(tf.random_normal([n_classes]), name='biases')

	pred = rnn_model(x, weights, biases)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	"""

	with tf.Session() as sess:
		rnn = ImageRNN(n_input, n_steps, n_hidden, n_classes)

		sess.run(tf.initialize_all_variables())
		step = 1

		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.all_variables())

		optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(rnn.cost)

		while step * params['batch_size'] < params['training_iters']:
			batch_x, batch_y = mnist.train.next_batch(params['batch_size'])
			# Reshape data to get 28 seq of 28 elements
			# batch_x = batch_x.reshape((params['batch_size'], n_steps, n_input))
			batch_x = tf.reshape(batch_x, (params['batch_size'], n_steps, n_input))
			sess.run(optimizer, feed_dict={rnn.input_x: batch_x, rnn.input_y: batch_y})
			sys.exit()
			if step % params['display_step'] == 0:
				path = saver.save(sess, checkpoint_prefix, global_step=step)
				acc = sess.run(rnn.accuracy, feed_dict={rnn.input_x: batch_x, rnn.input_y: batch_y})
				loss = sess.run(rnn.cost, feed_dict={rnn.input_x: batch_x, rnn.input_y: batch_y})
				print("Iter " + str(step * params['batch_size']) + ", Loss= " + "{:.6f}".format(loss) + ", Accuracy= " + "{:.5f}".format(acc))
			step += 1
		print("Optimization Finished!")

		test_len = 128
		test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
		test_label = mnist.test.labels[:test_len]
		print("Testing Accuracy:", sess.run(rnn.accuracy, feed_dict={rnn.input_x: test_data, rnn.input_y: test_label}))

if __name__ == '__main__':
	train()
