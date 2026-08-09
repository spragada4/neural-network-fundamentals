import random
random.seed(42)

from nn import MLP
from value import Value

# Toy dataset: 4 examples, 3 inputs each, binary-ish targets
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]   # desired outputs

model = MLP(n_inputs=3, layer_sizes=[4, 4, 1])

learning_rate = 0.05

for step in range(50):
    # ---- forward pass ----
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # ---- backward pass ----
    for p in model.parameters():
        p.grad = 0.0        # reset gradients (they accumulate otherwise!)
    loss.backward()

    # ---- update weights: move each param a little AGAINST its gradient ----
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if step % 10 == 0 or step == 49:
        print(f"step {step:2d}  loss = {loss.data:.4f}")

print("\nfinal predictions:", [round(model(x).data, 3) for x in xs])
print("targets:          ", ys)