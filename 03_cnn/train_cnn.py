import numpy as np
import sys
sys.path.append("../02_mlp")
from mnist_loader import load_mnist
from cnn_model import SimpleCNN

X_train, y_train_labels, X_test, y_test_labels = load_mnist()

# Reshape flat 784 vectors back into 28x28 images with 1 channel: (N, 1, 28, 28)
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
y_train = np.eye(10)[y_train_labels]

# CNNs are much slower than MLPs on CPU with our loop-based backward pass --
# use a smaller subset so this finishes in reasonable time
n_train = 5000
n_test = 1000
X_train, y_train = X_train[:n_train], y_train[:n_train]
X_test, y_test_labels = X_test[:n_test], y_test_labels[:n_test]

model = SimpleCNN(seed=0)

batch_size = 32
n_epochs = 3

for epoch in range(n_epochs):
    perm = np.random.permutation(n_train)
    X_train, y_train = X_train[perm], y_train[perm]

    for i in range(0, n_train, batch_size):
        Xb = X_train[i:i + batch_size]
        yb = y_train[i:i + batch_size]
        pred = model.forward(Xb)
        model.backward(yb, lr=0.05)

    train_pred = model.forward(X_train[:500])
    train_loss = model.loss(train_pred, y_train[:500])

    test_pred = model.forward(X_test)
    test_acc = np.mean(np.argmax(test_pred, axis=1) == y_test_labels)

    print(f"epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")