import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(float)


def softmax(z):
    # subtract max for numerical stability
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class MLP:
    def __init__(self, layer_sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialization: keeps activation variance stable across layers
            self.W.append(rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))

    def forward(self, X):
        self.z, self.a = [], [X]   # cache for backprop
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            is_last = (i == len(self.W) - 1)
            a = softmax(z) if is_last else relu(z)
            self.z.append(z)
            self.a.append(a)
        return a

    def backward(self, y_true, lr):
        m = y_true.shape[0]
        # gradient of softmax + cross-entropy loss combined simplifies to (pred - true)
        dz = self.a[-1] - y_true

        for i in reversed(range(len(self.W))):
            a_prev = self.a[i]
            dW = a_prev.T @ dz / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                da_prev = dz @ self.W[i].T
                dz = da_prev * relu_deriv(self.z[i - 1])

            self.W[i] -= lr * dW
            self.b[i] -= lr * db

    def loss(self, y_pred, y_true):
        eps = 1e-9
        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))