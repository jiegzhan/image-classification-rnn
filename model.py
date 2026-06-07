import tensorflow as tf

N_STEPS = 28  # rows per image (timesteps)
N_INPUT = 28  # pixels per row (features per timestep)
N_CLASSES = 10  # digits 0-9


def build_model(n_hidden=128, n_classes=N_CLASSES, learning_rate=0.001):
	"""Build and compile an LSTM classifier for MNIST rows-as-sequences."""
	model = tf.keras.Sequential([
		tf.keras.layers.Input(shape=(N_STEPS, N_INPUT)),
		tf.keras.layers.LSTM(n_hidden),
		tf.keras.layers.Dense(n_classes),
	])
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=["accuracy"],
	)
	return model
