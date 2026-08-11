import numpy as np
from mlp_numpy import MLP
import urllib.request
import gzip
import os

DATA_DIR = "mnist_data"
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def download():
    for name, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(BASE_URL + fname, path)


def load_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


download()
X_train = load_images(os.path.join(DATA_DIR, FILES["train_images"]))
y_train_labels = load_labels(os.path.join(DATA_DIR, FILES["train_labels"]))
X_test = load_images(os.path.join(DATA_DIR, FILES["test_images"]))
y_test_labels = load_labels(os.path.join(DATA_DIR, FILES["test_labels"]))

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