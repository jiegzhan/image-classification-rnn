### Project: Build an Image Classifier with RNN (LSTM) on TensorFlow

### Highlights

- This is a **multi-class image classification** problem.
- The purpose of this project is to **classify the MNIST image dataset into 10 classes**.
- The model is built with a **Recurrent Neural Network (RNN: LSTM)** on **TensorFlow 2.x / Keras**.
- Each 28×28 image is treated as **28 timesteps × 28 features** (one row per timestep).

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

- MNIST is downloaded automatically on first run via `tf.keras.datasets.mnist`.

### Train

```bash
python3 train.py ./training_parameters.json
```

A directory `trained_model_<timestamp>/` is created during training. The model is saved as `model.keras` inside that directory.

| Key | Default | Description |
|-----|---------|-------------|
| `learning_rate` | 0.001 | Adam learning rate |
| `training_iters` | 10000 | Approximate number of training samples to process |
| `batch_size` | 64 | Mini-batch size |
| `display_step` | 10 | Print progress every N epochs |

### Predict

Provide the model directory (created when running `train.py`) to `predict.py`:

```bash
python3 predict.py ./trained_model_<timestamp>/
```

### Reference

- [Recurrent Neural Network (TensorFlow-Examples)](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py) — original pattern this repo modernized for TensorFlow 2.x.
