import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.utils import shuffle

class DenseLayer:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))
        self.learning_rate = learning_rate

    def forward_propagate(self, inputs):
        self.inputs = inputs
        return self.relu(np.dot(self.weights, inputs) + self.biases)

    def backward_propagate(self, output_error):
        d_relu = self.d_relu(self.inputs)
        mat1 = np.mat(output_error)
        mat2 = np.mat(d_relu)
        if mat1.shape[1] != mat2.shape[0]:
            mat1 = mat1.T
        inputs_error = np.dot(mat1 * d_relu, self.weights)
        self.update_weights_biases(output_error, d_relu)
        return inputs_error

    def update_weights_biases(self, output_error, d_relu):
        self.weights -= np.dot(self.inputs, np.mat(output_error) * np.mat(d_relu)) * self.learning_rate
        self.biases -= np.sum(output_error * d_relu, axis=0) * self.learning_rate

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        return (x > 0).astype(float)
    
def mse_loss(y_true, y_pred):
    diff = y_true - y_pred
    return np.mean(diff**2), diff

class Model:  
    def __init__(self, layers):
        self.layers = layers
  
    def forward_propagate(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_propagate(inputs)
        return inputs

    def calculate_loss(self, y_true, y_pred):
        return mse_loss(y_true, y_pred)

    def backward_propagate(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward_propagate(delta)
            
    def train(self, X_train, Y_train, epochs=20, batch_size=4):
        for epoch in range(epochs):
            total_loss = self.process_batches(X_train, Y_train, batch_size)
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1} out of {epochs}, loss: {total_loss / len(X_train)}")

    def process_batches(self, X_train, Y_train, batch_size):
        total_loss = 0
        shuffle_idx = np.random.permutation(len(X_train))
        for i in shuffle_idx:
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]
            y_pred = self.forward_propagate(X_batch.T)
            loss, delta = self.calculate_loss(Y_batch.T, y_pred)
            self.backward_propagate(delta)
            total_loss += loss
        return total_loss
            
if __name__ == "__main__":
    np.random.seed(6)
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0  # normalize data
    Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)  # one-hot encoding
    X_train, X_test = X_train.reshape((-1, 784)), X_test.reshape((-1, 784))  # flatten data
    model = Model([DenseLayer(784, 128), DenseLayer(128, 10)])
    model.train(X_train, Y_train, epochs=20)
