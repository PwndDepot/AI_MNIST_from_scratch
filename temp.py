from dataclasses import dataclass
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np


@dataclass
class ActivationFunctions(object):
    @staticmethod
    def sigmoid(input):
        return 1 / (1 + np.exp(-input))

    @staticmethod
    def sigmoid_derivative(input):
        return (1 / (1 + np.exp(-input))) * (1 - (1 / (1 + np.exp(-input))))
    
    @staticmethod
    def softmax(input):
        return np.exp(input) / np.sum(np.exp(input))
    
    @staticmethod
    def relu(input):
        return np.maximum(0, input)

    @staticmethod
    def relu_derivative(input):
        return np.where(input <= 0, 0, 1)
    
@dataclass
class LossFunctions(object):
    @staticmethod
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_loss_derivative(y_true, y_pred):
        return y_pred - y_true
    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        outputs_softmax = np.exp(y_true) / np.sum(np.exp(y_true)) # same as ActivationFunction method sigmoid() but lets keep the classes separate (good programming practice)
        loss = -np.sum(y_pred * np.log(outputs_softmax))
        return loss
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_true = np.clip(y_true, epsilon, 1 - epsilon)
        return -np.mean(y_pred * np.log(y_true) + (1 - y_pred) * np.log(1 - y_true))

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_true = np.clip(y_true, epsilon, 1 - epsilon)
        return -np.sum(y_pred * np.log(y_true)) / y_true.shape[0]

class TwoLayerDenseNN:
    def __init__(self, input_size, hidden_size, output_size, activation_functions_L1=(ActivationFunctions.relu, ActivationFunctions.relu_derivative), 
                 activation_functions_L2=(ActivationFunctions.sigmoid, ActivationFunctions.sigmoid),
                 loss_functions_L1=(LossFunctions.mse_loss, LossFunctions.mse_loss_derivative), loss_functions_L2=(LossFunctions.mse_loss, LossFunctions.mse_loss_derivative), optimizer = None, learning_rate=0.1, epochs=100):
        # randomize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.hidden_size = hidden_size
        self.activation_function_L1 = activation_functions_L1[0] # using .get() rather than the indexing operator will return None instead of throwing most exceptions. Though sometimes that is 
                                                           # not always a good thing or what we want, so we'll grab the data via index to prevent a possible confusing exception later on.
        self.activation_function_derivative_L1 = activation_functions_L1[1]
        self.activation_function_L2 = activation_functions_L2[0]
        self.activation_function_derivative_L2 = activation_functions_L2[1]
        self.loss_function_L1 = loss_functions_L1[0]
        self.loss_function_derivative_L1 = loss_functions_L1[1]
        self.loss_function_L2 = loss_functions_L2[0]
        self.loss_function_derivative_L2 = loss_functions_L2[1]
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def one_hot(y):
        one_hot_y = np.zeros(y.size, y.max() + 1)
        one_hot_y[np.arange(y.size), y] = 1
        return one_hot_y.T

    def forward_propagation(self, X_train):
        """
        Perform forward propagation of the neural network.

        Parameters:
        X (ndarray): Input data of shape (number of examples, input_size)
        W1 (ndarray): Weights for the first layer of shape (input_size, hidden_size)
        b1 (ndarray): Biases for the first layer of shape (1, hidden_size)
        W2 (ndarray): Weights for the second layer of shape (hidden_size, output_size)
        b2 (ndarray): Biases for the second layer of shape (1, output_size)

        Returns:
        A2 (ndarray): The softmax output of the second activation
        cache (dict): A dictionary containing "Z1", "A1", "Z2" for backpropagation
        """
        # first dense layer
        Z1 = np.dot(X_train, self.W1) + self.b1 # Z1 = XW1 + b1
        A1 = self.activation_function_L1(Z1)

        # second dense layer
        Z2 = np.dot(A1, self.W2) + self.b2 # + self.b2 <-- this was causing an exception of different shape sizes
        A2 = self.activation_function_L2(Z2) # output

        # Cache the values for backpropagation
        cache = {
            "X": X_train,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2
        }
        return A2, cache
    
    """ def forward(self, X_train):
        # Forward pass through the network
        Z1 = np.dot(X_train, self.W1) + self.b1
        A1 = ActivationFunctions.relu(self.hidden_size)
        Z2 = np.dot(self.hidden_size, self.W2) + self.b2
        output = ActivationFunctions.relu
        cache = {
            #"X": X_train,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": output
        }
        return output, cache """

    def backward(self, X, y_true, y_pred, learning_rate):
        # Calculate gradient for W2 and b2
        delta2 = y_pred
        delta2[range(y_true.shape[0]), y_true] -= 1
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        # Calculate gradient for W1 and b1
        delta1 = np.dot(delta2, self.W2.T) * ActivationFunctions.sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)

        # Update the weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    """def backward(self, A1, A2, Y):
        m = Y.size
        one_hot_y = self.one_hot(Y)
        dZ2 = A2 - one_hot_y
        dW2 = 1 / m * self.dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(self.dZ2, 2)
        dZ1 = self.W2.T.dot(self.dZ2) # Update weights in reverse
        return dZ2, dW2, db2, dZ1"""
    
    def calculate_accuracy(self, A2, Y):
        predictions = np.argmax(A2, axis=1)
        labels = np.argmax(Y, axis=1)
        correct_predictions = np.sum(predictions == labels)
        total_predictions = Y.shape[0]
        accuracy = correct_predictions / total_predictions
        return accuracy

    def backward_propagation(self, cache, Y):
        """
        Perform backpropagation to compute gradients.

        Parameters:
        cache (dict): A dictionary containing "Z1", "A1", "Z2", "A2" from forward propagation
        Y (ndarray): True labels of shape (number of examples, output_size)

        Returns:
        grads (dict): A dictionary containing the gradients with respect to different parameters
        """
        # Retrieve values from cache
        A1 = cache['A1']
        A2 = cache['A2']
        
        # Compute the gradient of the loss function with respect to A2
        dLoss_dA2 = self.loss_function_derivative_L2(A2, Y)
        
        # Compute the gradient of the loss function with respect to Z2
        dLoss_dZ2 = dLoss_dA2 * self.activation_function_derivative_L2(cache['Z2'])
        
        # Compute the gradient of the loss function with respect to W2 and b2
        dLoss_dW2 = np.dot(A1.T, dLoss_dZ2)
        dLoss_db2 = np.sum(dLoss_dZ2, axis=0, keepdims=True)
        
        # Compute the gradient of the loss function with respect to A1
        dLoss_dA1 = np.dot(dLoss_dZ2, self.W2.T)
        
        # Compute the gradient of the loss function with respect to Z1
        dLoss_dZ1 = dLoss_dA1 * self.activation_function_derivative_L1(cache['Z1'])
        
        # Compute the gradient of the loss function with respect to W1 and b1
        #dLoss_dW1 = np.dot(cache['X'].T, dLoss_dZ1)
        dLoss_dW1 = np.dot(A2.T, dLoss_dZ1)
        dLoss_db1 = np.sum(dLoss_dZ1, axis=0, keepdims=True)
        
        return dLoss_dW1, dLoss_db1, dLoss_dW2, dLoss_db2
        # Store gradients in a dictionary
        '''grads = {
            "dW1": dLoss_dW1,
            "db1": dLoss_db1,
            "dW2": dLoss_dW2,
            "db2": dLoss_db2
        }'''
        
        return grads

    def update_parameters(self, dW1, db1, dW2, db2):
        # Gradients is a dictionary containing gradients of weights and biases
        # For example: gradients = {"dw1": ..., "db1": ..., "dw2": ..., "db2": ...}

        # Update weights and biases for each layer
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    """def gradient_descent(X, Y, interations, alpha):
        for i in range(interations):
            self.backward(X, )"""

    """ def get_accuracy(self, predictions, y):
        return np.sum(predictions == y) / y.size """
    
    def compute_loss(self, A2, Y):
        """
        Computes the cross-entropy loss.

        Parameters:
        A2 (ndarray): Output of the last layer of the neural network (after softmax), shape (number of examples, number of classes)
        Y (ndarray): True labels, shape (number of examples, number of classes)

        Returns:
        cost (float): Cross-entropy cost
        """
        m = Y.shape[0]  # number of examples

        # Compute the cross-entropy cost
        logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
        cost = - np.sum(logprobs) / m

        cost = np.squeeze(cost)  # makes sure cost is the dimension we expect (e.g., turns [[17]] into 17)
        return cost

    def train(self, X_train, Y_train, epochs):
        for epoch in range(epochs):
            #for _ in range(X_train.shape[0]): # Gradient Descent
            #X_i = X[i:i+1]
            #y_i = y[i:i+1]
            #z1, a1, z2, a2 = self.forward(X_i)
            output, cache = self.forward_propagation(X_train)
            loss = self.compute_loss(output, Y_train)
            accuracy = self.calculate_accuracy(output, Y_train)
            dLoss_dW1, dLoss_db1, dLoss_dW2, dLoss_db2 = self.backward_propagation(cache, Y_train)
            self.update_parameters(dLoss_dW1, dLoss_db1, dLoss_dW2, dLoss_db2)
            
            # Print the loss and accuracy
            y_pred = self.forward(X_train)
            #accuracy = self.get_accuracy(y_pred, Y_train)
            loss = LossFunctions.cross_entropy(Y_train, y_pred)
            print(f'Epoch {epoch}, loss: {loss}, accuracy: {accuracy}')

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

# Initialize the neural network with the size of the input layer, hidden layer, and output layer.
input_size = 784  # For MNIST, each image is 28x28 pixels
hidden_size = 64  # This can be adjusted
output_size = 10  # MNIST has 10 classes (digits 0-9)

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the images
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Normalize the data
X_train_flattened = X_train_flattened.astype('float32') / 255.0
X_test_flattened = X_test_flattened.astype('float32') / 255.0

# Convert labels to one-hot encoding using the custom function
y_train_encoded = to_categorical(y_train, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

nn = TwoLayerDenseNN(input_size=784, hidden_size=64, output_size=10, learning_rate=0.1)
nn.train(X_train_flattened, y_train_encoded, epochs=10)
predictions = nn.predict(X_test)
print(str(predictions))