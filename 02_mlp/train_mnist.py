import numpy as np
from mlp_numpy import MLP
from mnist_loader import load_mnist

X_train, y_train_labels, X_test, y_test_labels = load_mnist()
y_train = np.eye(10)[y_train_labels]

model = MLP(layer_sizes=[784, 128, 10], seed=0)

batch_size = 64
n_epochs = 5
n_samples = X_train.shape[0]

for epoch in range(n_epochs):
    perm = np.random.permutation(n_samples)
    X_train, y_train = X_train[perm], y_train[perm]

    for i in range(0, n_samples, batch_size):
        Xb = X_train[i:i + batch_size]
        yb = y_train[i:i + batch_size]
        pred = model.forward(Xb)
        model.backward(yb, lr=0.1)

    train_pred = model.forward(X_train[:2000])
    train_loss = model.loss(train_pred, y_train[:2000])

    test_pred = model.forward(X_test)
    test_acc = np.mean(np.argmax(test_pred, axis=1) == y_test_labels)

    print(f"epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")