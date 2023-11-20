from dataclasses import dataclass
from keras.datasets import mnist
from keras.utils import to_categorical
import time
import numpy as np


@dataclass
class ActivationFunctions(object):
    @staticmethod
    def sigmoid(input, derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-input))
        else:
            return (1 / (1 + np.exp(-input))) * (1 - (1 / (1 + np.exp(-input))))
    
    @staticmethod
    def softmax(input):
        return np.exp(input) / np.sum(np.exp(input))
    
    @staticmethod
    def relu(input, derivative=False):
        if not derivative:
            return np.maximum(0, input)
        else:
            return np.where(input <= 0, 0, 1)
    
@dataclass
class LossFunctions(object):
    @staticmethod
    def mse_loss(y_true, y_pred, derivative=False):
        if not derivative:
            return np.mean((y_true - y_pred) ** 2)
        else:
            return y_pred - y_true
        
    @staticmethod
    def cross_entropy(y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / m
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

@dataclass
class Optimizers(object):
    # incomplete
    @staticmethod
    def SGD(gradient_weights, gradient_bias, learning_rate=0.001):
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
        return weights, bias
    
    @staticmethod
    def Adam(params, grads, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if m is None or v is None:  # initialize m and v
            m = [np.zeros_like(param) for param in params]
            v = [np.zeros_like(param) for param in params]

        t += 1
        new_params = []

        for param, grad, m_i, v_i in zip(params, grads, m, v):
            m_i = beta1 * m_i + (1 - beta1) * grad
            v_i = beta2 * v_i + (1 - beta2) * (grad ** 2)

            m_hat = m_i / (1 - beta1 ** t)
            v_hat = v_i / (1 - beta2 ** t)

            param_update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            param -= param_update

            new_params.append(param)

        return new_params, m, v, t

class TwoLayerDenseNN(object):
    def __init__(self, input_size, hidden_size, output_size, activation_function_L1 = ActivationFunctions.relu, 
                 activation_functions_L2 = (ActivationFunctions.relu, ActivationFunctions.sigmoid), loss_function_L1 = LossFunctions.mse_loss,
                             loss_function_L2 = LossFunctions.mse_loss, optimizer = Optimizers.Adam, adam_optimizations = None, learning_rate=0.001):
        self.X_train = None
        self.Y_train = None
        # randomize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.activation_function1_L1 = activation_function_L1 # using .get() rather than the indexing operator will return None instead of throwing most exceptions. Though sometimes that is 
                                                           # not always a good thing or what we want, so we'll grab the data via index to prevent a possible confusing exception later on.
        self.activation_function1_L2 = activation_functions_L2[0]
        self.activation_function2_L2 = activation_functions_L2[1]
        self.loss_function_L1 = loss_function_L1
        self.loss_function_L2 = loss_function_L2
        self.optimizer = optimizer
        self.learning_rate = learning_rate
    
    def one_hot_encode(self, y, num_classes=10):
        y = np.array(y, dtype=int)  # Ensure y is an array of integers
        return np.eye(num_classes)[y]

    def forward_propagation(self, X):
        # first dense layer
        Z1 = (np.dot(X, self.W1) + self.b1) # Z1 = XW1 + b1
        A1 = self.activation_function1_L1(Z1)

        # second dense layer
        Z2 = (np.dot(A1, self.W2) + self.b2)
        A2 = self.activation_function2_L2(Z2) # output
        return A2, Z1, A1, Z2
    
    def calculate_accuracy(self, X, Y):
        predictions = np.argmax(X, axis=1)
        labels = np.argmax(Y, axis=1)
        correct_predictions = np.sum(predictions == labels)
        total_predictions = Y.shape[0]
        accuracy = correct_predictions / total_predictions
        return accuracy

    def backward_propagation(self, X, Y, A2, Z1, A1, Z2):
        # compute the gradient of the loss function with respect to A2
        dLoss_dA2 = self.loss_function_L2(A2, Y, derivative=True)
        
        # compute the gradient of the loss function with respect to Z2
        dLoss_dZ2 = dLoss_dA2 * self.activation_function2_L2(Z2, derivative=True)
        
        # compute the gradient of the loss function with respect to W2 and b2
        dLoss_dW2 = np.dot(A1.T, dLoss_dZ2)
        dLoss_db2 = np.sum(dLoss_dZ2, axis=0, keepdims=True)
        
        # compute the gradient of the loss function with respect to A1
        dLoss_dA1 = np.dot(dLoss_dZ2, self.W2.T)
        
        # compute the gradient of the loss function with respect to Z1
        dLoss_dZ1 = dLoss_dA1 * self.activation_function1_L2(Z1, derivative=True)
        
        # compute the gradient of the loss function with respect to W1 and b1
        dLoss_dW1 = np.dot(X.T, dLoss_dZ1)
        #dLoss_dW1 = np.dot(A2.T, dLoss_dZ1)
        dLoss_db1 = np.sum(dLoss_dZ1, axis=0, keepdims=True)
        
        return dLoss_dW1, dLoss_db1, dLoss_dW2, dLoss_db2

    def update_parameters(self, dW1, db1, dW2, db2):
        # update weights and biases for each layer
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def compute_loss(self, X, Y):
        m = Y.shape[0]  # number of examples

        # compute the cross-entropy cost
        logprobs = np.multiply(np.log(X), Y) + np.multiply((1 - Y), np.log(1 - X))
        cost = - np.sum(logprobs) / m

        cost = np.squeeze(cost)  # makes sure cost is the dimension we expect (e.g., turns [[17]] into 17)
        return cost
    
    def predict(self, y_true, y_pred):
        correct_predictions = sum(y_pred == y_true)
        total_predictions = len(y_true)
        accuracy = correct_predictions / total_predictions
        return accuracy

    def train(self, X_train, Y_train, epochs):
        self.X_train = X_train
        self.Y_train = Y_train

        for epoch in range(epochs):
            output, Z1, A1, Z2 = self.forward_propagation(self.X_train)

            loss = LossFunctions.cross_entropy(self.Y_train, output)
            accuracy = self.calculate_accuracy(output, self.Y_train)
            
            dLoss_dW1, dLoss_db1, dLoss_dW2, dLoss_db2 = self.backward_propagation(self.X_train, self.Y_train, output, Z1, A1, Z2)
            if self.optimizer is None:
                self.update_parameters(dLoss_dW1, dLoss_db1, dLoss_dW2, dLoss_db2)
            else:
                if self.optimizer is Optimizers.Adam:
                    params = [self.W1, self.b1, self.W2, self.b2]
                    gradients = [dLoss_dW1, dLoss_db1, dLoss_dW2, dLoss_db2]
                    m_opt = [np.zeros_like(param) for param in params]
                    v_opt = [np.zeros_like(param) for param in params]
                    t_opt = 0
                    self.optimizer(params, gradients, m_opt, v_opt, t_opt, learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-15)

            print(f'Epoch {epoch}, loss: {loss}, accuracy: {accuracy}')

class DenseNN(object):
    def __init__(self, n_inputs, m_outputs):
        self.n_inputs = n_inputs
        self.m_outputs = m_outputs


# globals
EPOCHS = 10
LEARNING_RATE = 0.001
LOSS_FUNCTION_L1 = LossFunctions.mse_loss
LOSS_FUNCTION_L2 = LossFunctions.mse_loss
ACTIVATION_FUNCTION_L1 = ActivationFunctions.relu
ACTIVATION_FUNCTION_L2 = ActivationFunctions.sigmoid

# optimizer parameters
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-15

# initialize the neural network with the size of the input layer, hidden layer, and output layer.
input_size = 28 * 28  # for MNIST, each image is 28x28 pixels
hidden_size = 64  # arbitrary number
output_size = 10  # MNIST has 10 classes (numbers 0-9)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# preprocess images
# flatten
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# normalize the data
X_train_flattened = X_train_flattened.astype('float32') / 255.0
X_test_flattened = X_test_flattened.astype('float32') / 255.0

# convert labels to one-hot encoding using the custom function
y_train_encoded = to_categorical(y_train, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

nn = TwoLayerDenseNN(input_size=784, hidden_size=64, output_size=10, learning_rate=LEARNING_RATE)
nn.train(X_train_flattened, y_train_encoded, epochs=EPOCHS)
#predictions = nn.predict(X_test_flattened, y_test_encoded)
#print(str(predictions))



quit()

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
keras_model.compile(optimizer=SGD(learning_rate=LEARNING_RATE), loss='mean_squared_error', metrics=['accuracy'])

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