import json
import math
import os
import sys
import time

import tensorflow as tf

from model import N_CLASSES, build_model


DEFAULT_PARAMS = {
	"learning_rate": 0.001,
	"training_iters": 10000,
	"batch_size": 64,
	"display_step": 10,
	"n_hidden": 128,
	"test_samples": 500,
}


def load_params(parameter_file):
	with open(parameter_file) as f:
		params = json.load(f)
	return {**DEFAULT_PARAMS, **params}


def load_mnist():
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train = x_train.astype("float32") / 255.0
	x_test = x_test.astype("float32") / 255.0
	return x_train, y_train, x_test, y_test


class ProgressCallback(tf.keras.callbacks.Callback):
	def __init__(self, batch_size, display_step):
		super().__init__()
		self.batch_size = batch_size
		self.display_step = display_step

	def on_epoch_end(self, epoch, logs=None):
		if (epoch + 1) % self.display_step != 0:
			return
		samples_seen = (epoch + 1) * self.batch_size
		loss = logs.get("loss", 0.0)
		accuracy = logs.get("accuracy", 0.0)
		print(
			"Iter: {}, Loss: {:.6f}, Accuracy: {:.6f}".format(
				samples_seen, loss, accuracy
			)
		)


def train():
	if len(sys.argv) < 2:
		print("Usage: python train.py training_parameters.json")
		sys.exit(1)

	params = load_params(sys.argv[1])
	x_train, y_train, x_test, y_test = load_mnist()

	model = build_model(
		n_hidden=params["n_hidden"],
		n_classes=N_CLASSES,
		learning_rate=params["learning_rate"],
	)

	epochs = math.ceil(params["training_iters"] / params["batch_size"])
	timestamp = str(int(time.time()))
	out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
	os.makedirs(out_dir, exist_ok=True)
	model_path = os.path.join(out_dir, "model.keras")

	model.fit(
		x_train,
		y_train,
		batch_size=params["batch_size"],
		epochs=epochs,
		steps_per_epoch=1,
		validation_data=(x_test, y_test),
		verbose=0,
		callbacks=[ProgressCallback(params["batch_size"], params["display_step"])],
	)

	model.save(model_path)
	print("The training is done")
	print("Saved model to:", model_path)

	test_len = min(params["test_samples"], len(x_test))
	loss, accuracy = model.evaluate(x_test[:test_len], y_test[:test_len], verbose=0)
	print("Testing Accuracy:", accuracy)


if __name__ == "__main__":
	train()
