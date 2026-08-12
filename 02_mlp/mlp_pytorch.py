import torch
import torch.nn as nn
import numpy as np
from mnist_loader import load_mnist

torch.manual_seed(0)

X_train, y_train_labels, X_test, y_test_labels = load_mnist()

# Convert to torch tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train_labels, dtype=torch.long)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test_labels, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Same shape as your NumPy version: 784 -> 128 -> 10
        self.fc1 = nn.Linear(784, 128)   # replaces your W[0], b[0]
        self.fc2 = nn.Linear(128, 10)    # replaces your W[1], b[1]

    def forward(self, x):
        x = torch.relu(self.fc1(x))      # same relu() you wrote by hand
        x = self.fc2(x)                  # no softmax here -- see note below
        return x


model = MLP()
# CrossEntropyLoss expects raw scores (logits) and applies softmax internally,
# combined with log for numerical stability -- this is the PyTorch equivalent
# of the "softmax + cross-entropy, gradient simplifies to (pred - true)" trick
# you implemented by hand in mlp_numpy.py's backward().
criterion = nn.CrossEntropyLoss()
# SGD is literally your "W -= lr * dW" loop, just done for you
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

batch_size = 64
n_epochs = 5
n_samples = X_train.shape[0]

for epoch in range(n_epochs):
    perm = torch.randperm(n_samples)
    X_train, y_train = X_train[perm], y_train[perm]

    for i in range(0, n_samples, batch_size):
        Xb = X_train[i:i + batch_size]
        yb = y_train[i:i + batch_size]

        optimizer.zero_grad()            # same as your "p.grad = 0.0" reset
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()                  # autograd -- what your Value.backward() did by hand
        optimizer.step()                 # same as your "W -= lr * dW"

    with torch.no_grad():
        train_logits = model(X_train[:2000])
        train_loss = criterion(train_logits, y_train[:2000]).item()

        test_logits = model(X_test)
        test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

    print(f"epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")