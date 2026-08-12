import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("../02_mlp")
from mnist_loader import load_mnist

torch.manual_seed(0)

X_train, y_train_labels, X_test, y_test_labels = load_mnist()
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

# Match the NumPy version's data subset for a fair comparison
n_train, n_test = 5000, 1000
X_train, y_train_labels = X_train[:n_train], y_train_labels[:n_train]
X_test, y_test_labels = X_test[:n_test], y_test_labels[:n_test]

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train_labels, dtype=torch.long)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test_labels, dtype=torch.long)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Same shape as your from-scratch version: 1->8 channels, 3x3 kernel
        self.conv = nn.Conv2d(1, 8, kernel_size=3, stride=1)   # replaces your Conv2D
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)      # replaces your MaxPool2D
        self.fc = nn.Linear(8 * 13 * 13, 10)                   # replaces your W_fc, b_fc

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)   # flatten, same as your self.flat
        x = self.fc(x)
        return x


model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

batch_size = 32
n_epochs = 3

for epoch in range(n_epochs):
    perm = torch.randperm(n_train)
    X_train, y_train = X_train[perm], y_train[perm]

    for i in range(0, n_train, batch_size):
        Xb = X_train[i:i + batch_size]
        yb = y_train[i:i + batch_size]

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = model(X_train[:500])
        train_loss = criterion(train_logits, y_train[:500]).item()
        test_logits = model(X_test)
        test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

    print(f"epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")