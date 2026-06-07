import os
import sys

import tensorflow as tf

TEST_SAMPLES = 500


def load_mnist():
	(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_test = x_test.astype("float32") / 255.0
	return x_test, y_test


def predict():
	if len(sys.argv) < 2:
		print("Usage: python predict.py ./trained_model_directory/")
		sys.exit(1)

	model_dir = os.path.abspath(sys.argv[1])
	model_path = os.path.join(model_dir, "model.keras")

	if not os.path.isfile(model_path):
		print("Model not found at:", model_path)
		sys.exit(1)

	print("Loaded the trained model:", model_path)
	model = tf.keras.models.load_model(model_path)

	x_test, y_test = load_mnist()
	test_len = min(TEST_SAMPLES, len(x_test))
	loss, accuracy = model.evaluate(x_test[:test_len], y_test[:test_len], verbose=0)
	print("Testing Accuracy:", accuracy)


if __name__ == "__main__":
	predict()
