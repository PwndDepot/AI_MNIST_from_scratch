from dataclasses import dataclass
from keras.datasets import mnist
from keras.utils import to_categorical
import time
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from ai_utils import ai_utils


class Model(object):
    def __init__(self, m_input_size, hidden_size, n_output_size, x_train, y_train, layers = None):
        if not layers:
            raise NotImplementedError("Nueral network incomplete (missing layers)")
        self.m_input_size = m_input_size
        self.hidden_size = hidden_size
        self.n_output_size = n_output_size
        self.x_train = x_train
        self.y_train = y_train
        self.layers = layers

    def forward_propagate(self, input_data, activation_function=ai_utils.ActivationFunctions.relu):
        output_layer1 = self.layers[0].forward_propagate(input_data, activation_function=ai_utils.ActivationFunctions.relu)
        output_layer2 = self.layers[1].forward_propagate(output_layer1, activation_function=ai_utils.ActivationFunctions.softmax)
        return output_layer2

    def backward_propagate(self, output_gradient, learning_rate, derivative_function=None):
        # backpropagate through the second layer
        error_layer2 = self.layers[1].backward_propagate(output_gradient, learning_rate) # output_error already has softmax gradient and cross_entropy gradient due to some neat trick I couldn't possibly explain
        # backpropagate through the first layer
        self.layers[0].backward_propagate(error_layer2, learning_rate, derivative_function=ai_utils.ActivationFunctions.relu_derivative)

    def fit(self, x_train, y_train, epochs = 50, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            for x, y_true in zip(x_train, y_train):
                if x.ndim == 1: x.shape += (1,)
                if x.ndim == 1: y_true += (1,)

                # forward propagation
                y_pred = self.forward_propagate(x)

                # calc loss using mse (or other)
                loss = ai_utils.LossFunctions.categorical_cross_entropy(y_true, y_pred)
                total_loss += loss

                # back prop and update weights
                output_error = y_true - y_pred
                gradient = y_pred - y_true # softmax cross_entropy gradient fancy trick
                self.backward_propagate(gradient, learning_rate)

                # calculate accuracy (for classification tasks)
                if np.argmax(output_error) == np.argmax(y_true):
                    correct_predictions += 1
                total_predictions += 1

                average_loss = float(float(loss) / float(total_loss))
                accuracy = float(float(correct_predictions) / (float(total_predictions)))
                if epoch % 20 == 1:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

                # backprop
            # average loss/accuracy for the epoch
            # i'll float everything watch me
            average_loss = float(float(total_loss) / float(len(x_train)))
            accuracy = float(float(correct_predictions) / float(len(x_train)))

            print(f"Epoch {epoch+1}/{epochs} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
        

class DenseLayer(object):
    def __init__(self, m_input_size, n_output_size):
        self.m_input_size = m_input_size
        self.n_output_size = n_output_size
        #self.weights = np.random.randn(m_input_size, n_output_size) * 0.01 # multiply by a small number so generated numbers are smaller, improves optimization when training
        self.weights = np.random.randn(n_output_size, m_input_size)
        self.biases = np.zeros((n_output_size, 1))

    def forward_propagate(self, input_data, activation_function=ai_utils.ActivationFunctions.relu):
        self.input = input_data
        # y = W * X + B
        pre_activation = np.dot(input_data, self.weights) + self.biases
        #activated = activation_function(pre_activation)
        return pre_activation
    
    def backward_propagate(self, output_gradient, learning_rate, derivative_function=None):
        if derivative_function: # skip softmax and cross_entropy gradient, but use relu derivative
            output_gradient = derivative_function(output_gradient)
        weights_gradient = np.dot(output_gradient, learning_rate)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient

        return np.dot(self.weights.T, output_gradient)

class Activation(DenseLayer):
    def __init__(self, x_train, activation_function):
        self.activation_function = activation_function
        self.A0 = x_train


    def forward(self, x_train):
        self.input = input
        return self.activation(self.input)
    
    def backword(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

# initialize the neural network with the size of the input layer, hidden layer, and output layer.
input_size = 28 * 28  # for MNIST, each image is 28x28 pixels
hidden_size = 64  # arbitrary number
output_size = 10  # MNIST has 10 classes (numbers 0-9)

# load data
(x_train, labels), (x_test, y_test) = mnist.load_data()

# preprocess images
#X_train_norm = X_train.reshape((X_train.shape[0], -1))
x_train = x_train.reshape(x_train.shape[0], input_size, 1)
#X_test_flattened = X_test.reshape((X_test.shape[0], -1))

# normalize the data
X_train_flattened = x_train.astype('float32') / 255
#X_test_flattened = X_test_flattened.astype('float32') / 255.0

# convert labels to one-hot encoding using the custom function
labels = to_categorical(labels, num_classes=10)
labels = labels.reshape(labels.shape[0], output_size, 1)
#y_test_encoded = to_categorical(y_test, num_classes=10)

layers = [DenseLayer(input_size, hidden_size), DenseLayer(hidden_size, output_size)]

model = Model(input_size, hidden_size, output_size, X_train_flattened, labels, layers)

model.fit(X_train_flattened, labels)