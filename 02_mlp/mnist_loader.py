import numpy as np
import urllib.request
import gzip
import os

DATA_DIR = "mnist_data"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
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


def load_mnist():
    download()
    X_train = load_images(os.path.join(DATA_DIR, FILES["train_images"]))
    y_train = load_labels(os.path.join(DATA_DIR, FILES["train_labels"]))
    X_test = load_images(os.path.join(DATA_DIR, FILES["test_images"]))
    y_test = load_labels(os.path.join(DATA_DIR, FILES["test_labels"]))
    return X_train, y_train, X_test, y_test