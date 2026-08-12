import numpy as np
import sys
sys.path.append("../02_mlp")
from conv_layer import Conv2D
from pool_layer import MaxPool2D


def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(float)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class SimpleCNN:
    """Conv -> ReLU -> MaxPool -> Flatten -> Dense -> Softmax"""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.conv = Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, seed=seed)
        self.pool = MaxPool2D(size=2, stride=2)

        # After conv: 28x28 -> 26x26 (8 channels). After pool: 26x26 -> 13x13.
        flat_size = 8 * 13 * 13
        self.W_fc = rng.normal(0, np.sqrt(2.0 / flat_size), size=(flat_size, 10))
        self.b_fc = np.zeros((1, 10))

    def forward(self, X):
        # X: (N, 1, 28, 28)
        self.conv_out = self.conv.forward(X)          # (N, 8, 26, 26)
        self.relu_out = relu(self.conv_out)
        self.pool_out = self.pool.forward(self.relu_out)  # (N, 8, 13, 13)

        N = X.shape[0]
        self.flat = self.pool_out.reshape(N, -1)        # (N, 8*13*13)
        logits = self.flat @ self.W_fc + self.b_fc       # (N, 10)
        self.probs = softmax(logits)
        return self.probs

    def backward(self, y_true, lr):
        N = y_true.shape[0]
        dlogits = (self.probs - y_true) / N              # softmax+CE gradient, same trick as MLP

        dW_fc = self.flat.T @ dlogits
        db_fc = np.sum(dlogits, axis=0, keepdims=True)
        dflat = dlogits @ self.W_fc.T

        self.W_fc -= lr * dW_fc
        self.b_fc -= lr * db_fc

        dpool_out = dflat.reshape(self.pool_out.shape)
        drelu_out = self.pool.backward(dpool_out)
        dconv_out = drelu_out * relu_deriv(self.conv_out)
        self.conv.backward(dconv_out, lr)   # updates conv weights internally

    def loss(self, y_pred, y_true):
        eps = 1e-9
        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))