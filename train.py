import os
import sys
import time
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
training_iters = 15000
batch_size = 128
display_step = 10

n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weights')
biases = tf.Variable(tf.random_normal([n_classes]), name='biases')

def RNN(x, weights, biases):
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, n_input])
	x = tf.split(0, n_steps, x)

	lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
	outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights) + biases

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
saver = tf.train.Saver(tf.all_variables())

with tf.Session() as sess:
	sess.run(init)
	step = 1

	timestamp = str(int(time.time()))
	out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
	checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
	checkpoint_prefix = os.path.join(checkpoint_dir, "model")
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	while step * batch_size < training_iters:
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		# Reshape data to get 28 seq of 28 elements
		batch_x = batch_x.reshape((batch_size, n_steps, n_input))
		# Run optimization op (backprop)
		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
		if step % display_step == 0:
			path = saver.save(sess, checkpoint_prefix, global_step=step)
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
			# Calculate batch loss
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
			print("Iter " + str(step*batch_size) + ", Loss= " + "{:.6f}".format(loss) + ", Accuracy= " + "{:.5f}".format(acc))
		step += 1
	print("Optimization Finished!")

	# Calculate accuracy for 128 mnist test images
	test_len = 128
	test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
	test_label = mnist.test.labels[:test_len]
	print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
