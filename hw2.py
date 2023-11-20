from dataclasses import dataclass
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
import numpy as np
import time


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
    def categorical_cross_entropy(predictions, targets):
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.sum(targets * np.log(predictions)) / predictions.shape[0]

class TwoLayerDNN(object):
    def __init__(self, n_inputs, n_hidden, m_outputs, activation_functions=(ActivationFunctions.relu, ActivationFunctions.relu_derivative), loss_functions=(LossFunctions.mse_loss, LossFunctions.mse_loss_derivative), optimizer = None, learning_rate=0.1, epochs=100):
        # randomize weights and biases
        self.w1 = np.random.randn(n_inputs, n_hidden) * 0.01
        self.b1 = np.zeros((1, n_hidden))
        self.w2 = np.random.randn(n_hidden, m_outputs) * 0.01
        self.b2 = np.zeros((1, n_hidden))
        self.activation_function = activation_functions[0] # using .get() rather than the indexing operator will return None instead of throwing most exceptions. Though sometimes that is 
                                                           # not always a good thing or what we want, so we'll grab the data via index to prevent a possible confusing exception later on.
        self.activation_function_derivative = activation_functions[1]
        self.loss_function = loss_functions[0]
        self.loss_function_derivative = loss_functions[1]
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def forward_propagation(self, X):
        """
        Perform forward propagation of the neural network.

        Parameters:
        X (ndarray): Input data of shape (number of examples, input_size)
        W1 (ndarray): Weights for the first layer of shape (input_size, hidden_size)
        b1 (ndarray): Biases for the first layer of shape (1, hidden_size)
        W2 (ndarray): Weights for the second layer of shape (hidden_size, output_size)
        b2 (ndarray): Biases for the second layer of shape (1, output_size)

        Returns:
        A2 (ndarray): The output of the second activation (default softmax)
        cache (dict): A dictionary containing "Z1", "A1", "Z2" for backpropagation
        """
        # first layer
        z1 = np.dot(X, self.w1) + self.b1
        a1 = self.activation_function(z1)

        # second layer
        z2 = np.dot(a1, self.w2)# + self.b2
        a2 = self.activation_function(z2)

        # save values for backpropagation
        cache = {
            "Z1": z1,
            "A1": a1,
            "Z2": z2,
            "A2": a2
        }
        return a2, cache
    
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
        dLoss_dA2 = self.loss_function_derivative(A2, Y)
        
        # Compute the gradient of the loss function with respect to Z2
        dLoss_dZ2 = dLoss_dA2 * self.activation_function_derivative(cache['Z2'])
        
        # Compute the gradient of the loss function with respect to W2 and b2
        dLoss_dW2 = np.dot(A1.T, dLoss_dZ2)
        dLoss_db2 = np.sum(dLoss_dZ2, axis=0, keepdims=True)
        
        # Compute the gradient of the loss function with respect to A1
        dLoss_dA1 = np.dot(dLoss_dZ2, self.w2.T)
        
        # Compute the gradient of the loss function with respect to Z1
        dLoss_dZ1 = dLoss_dA1 * self.activation_function_derivative(cache['Z1'])
        
        # Compute the gradient of the loss function with respect to W1 and b1
        dLoss_dW1 = np.dot(cache['X'].T, dLoss_dZ1)
        dLoss_db1 = np.sum(dLoss_dZ1, axis=0, keepdims=True)
        
        # Store gradients in a dictionary
        grads = {
            "dW1": dLoss_dW1,
            "db1": dLoss_db1,
            "dW2": dLoss_dW2,
            "db2": dLoss_db2
        }
        return grads

    def predict(self, x):
        """
        Get a prediction by running through the network once and using argmax on the output.
        """
        output = self.forward_propagation(self, x)
        return np.argmax(output)
    
    def compute_loss(A2, Y):
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
    
    def train(self, data, labels):
        for epoch in range(self.epochs):
            total_loss = 0
            correct_predictions = 0

            # loop through data and labels
            for x, y in zip(data, labels):
                # forward pass
                #output = self.forward_propagation(x)
                output, cache = self.forward_propagation(x)
                grads = self.backward_propagation(cache, y)
                
                # compute loss
                loss = self.loss_function(y, output)
                total_loss += loss

                # compute accuracy
                predicted_class = output > 0.5
                correct_predictions += int(predicted_class == y)
                
                # pdate weights
                if not self.optimizer:
                    # calculate gradients
                    gradients = self.compute_gradients(x, y, output)
                    # update weights with gradients
                    self.update_weights(gradients)
                else:
                    gradient_optimizer = self.loss_function_derivative(y, output)
                    self.optimizer.update_weights(self.parameters, gradient_optimizer)

            # calculate total accuracy
            accuracy = correct_predictions / len(data)

            # print the loss and accuracy
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(data)}, Accuracy: {accuracy*100:.2f}%")
    
@dataclass
class Optimizers(object):
    # incomplete
    def SGD(weights, bias, learning_rate):
        weights -= learning_rate * weights.gradient_weights
        bias -= learning_rate * bias.gradient_bias
        return weights, bias
    
    # missing adam optimizer and others...

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# flatten the images
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# convert labels to one-hot encoding using the custom function
y_train_encoded = to_categorical(y_train, num_classes=10).T
y_test_encoded = to_categorical(y_test, num_classes=10).T

# create function pointer tuples to pass to our custom model class constructor
activation_fp_tuple = (ActivationFunctions.relu, ActivationFunctions.relu_derivative)
loss_fp_tuple = (LossFunctions.mse_loss, LossFunctions.mse_loss_derivative)

# instantiate the custom neural network
nn = TwoLayerDNN(n_inputs=784, n_hidden=64, m_outputs=10, learning_rate=0.1, epochs=10, activation_functions=activation_fp_tuple, loss_functions=loss_fp_tuple, optimizer=Optimizers.SGD)

# train the custom neural network
start_time = time.time()
nn.train(X_train_flattened, y_train_encoded)
end_time = time.time()

# make predictions with the custom neural network
predictions = nn.predict(X_test_flattened, activation_function=nn.sigmoid)
predictions = np.argmax(predictions, axis=0)
true_labels = np.argmax(y_test_encoded, axis=0)

# calculate accuracy and loss for the custom neural network
custom_nn_accuracy = accuracy_score(true_labels, predictions)
custom_nn_time = end_time - start_time







###################### Keras ######################
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten the images
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Convert labels to one-hot encoding using the custom function
y_train_encoded = to_categorical(y_train, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

# Define the Keras model
keras_model = Sequential([
    Dense(64, input_shape=(784,), activation='sigmoid'),
    Dense(10, activation='sigmoid')
])

# Compile the Keras model
keras_model.compile(optimizer=SGD(learning_rate=0.1), loss='mean_squared_error', metrics=['accuracy'])

# Train the Keras model
start_time = time.time()
keras_history = keras_model.fit(X_train_flattened, y_train_encoded, epochs=10, batch_size=1, verbose=2)
end_time = time.time()

# Evaluate the Keras model
keras_loss, keras_accuracy = keras_model.evaluate(X_test, y_test_encoded, verbose=2)
keras_time = end_time - start_time

print(f"Keras Model Accuracy: {keras_accuracy}")
print(f"Keras Model Loss: {keras_loss}")
print(f"Keras Model Training Time: {keras_time} seconds")