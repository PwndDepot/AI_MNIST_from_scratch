from dataclasses import dataclass
import numpy as np


@dataclass
class ActivationFunctions(object):
    # this is a mess. tried to be fancy and it backfired
    @staticmethod
    def sigmoid(input, derivative=False, **kwargs):
        if not derivative:
            return 1 / (1 + np.exp(-input)) # y = 1 / (1 + e^-x)
        else:
            return (1 / (1 + np.exp(-input))) * (1 - (1 / (1 + np.exp(-input))))
    
    def softmax(input):
        e = np.exp(input - input.max(axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)
    
    @staticmethod
    def relu(input):
        return np.maximum(0, input)
    
    @staticmethod
    def relu_derivative(input):
        return np.where(input <= 0, 0, 1).astype('int')
    
@dataclass
class LossFunctions(object):
    @staticmethod
    def mse_loss(y_true, y_pred):
        n = len(y_true)
        mse_loss = np.sum((y_pred - y_true) ** 2) / n
        return mse_loss
        
    @staticmethod
    def cross_entropy(y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        return y_pred - y_true

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_true = np.clip(y_true, epsilon, 1 - epsilon)
        return -np.mean(y_pred * np.log(y_true) + (1 - y_pred) * np.log(1 - y_true))

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # calc for each sample
        loss = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        # average loss
        mean_loss = np.mean(loss)
        return mean_loss
    
    @staticmethod
    def categorical_cross_entropy_derivative(dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        # if labels sparse, one-hot em
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # calc gradient
        dinputs = -y_true / dvalues
        # normalize
        dinputs = dinputs / samples
        return dinputs

@dataclass
class Optimizers(object):
    # incomplete
    @staticmethod
    def SGD(gradient_weights, gradient_bias, learning_rate=0.001):
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
        return weights, bias
    
    def SGD(layer, weights, bights, dweights, dbiases, learning_rate=0.001):
        layer.weights += -learning_rate * layer.dweights
        layer.biases += -learning_rate * layer.dbiases
    
    @staticmethod
    # I have no idea how this works
    def Adam(params, grads, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
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
    