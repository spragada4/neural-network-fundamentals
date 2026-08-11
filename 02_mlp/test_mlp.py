import numpy as np
from mlp_numpy import MLP

np.random.seed(0)

# XOR problem: not linearly separable, needs a hidden layer to solve
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

# one-hot targets: class 0 = "XOR is 0", class 1 = "XOR is 1"
y_labels = np.array([0, 1, 1, 0])
y = np.eye(2)[y_labels]

model = MLP(layer_sizes=[2, 8, 2], seed=0)

for step in range(2000):
    pred = model.forward(X)
    loss = model.loss(pred, y)
    model.backward(y, lr=0.5)
    if step % 400 == 0:
        print(f"step {step:4d}  loss = {loss:.4f}")

final = model.forward(X)
print("\npredicted classes:", np.argmax(final, axis=1))
print("true classes:      ", y_labels)