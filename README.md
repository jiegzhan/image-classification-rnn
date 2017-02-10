### Project: Build an Image Classifier with RNN(LSTM) on Tensorflow

### Highlights:

 - This is a **multi-class image classification** problem.
 - The purpose of this project is to **classify MNIST image dataset into 10 classes**. 
 - The model was built with **Recurrent Neural Network (RNN: LSTM)** on **Tensorflow**.
 
### Data:

  - The MNIST image dataset was saved in the ./data/ directory.

### Train:

 - Command: python3 train.py parameters.json
 - Example: ```python3 train.py ./parameters.json```
 
 A directory will be created during training, and the model will be saved in this directory. 

### Predict:

 Provide the model directory (created when running ```train.py```) to ```predict.py```.
 - Command: python3 predict.py ./trained_model_directory/
 - Example: ```python3 predict.py ./trained_model_1481170507/```

### Reference:
 - [Recurrent Neural Network](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py)
