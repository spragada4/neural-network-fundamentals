import random
random.seed(42)

from nn import MLP

# 3 inputs -> hidden layer of 4 -> hidden layer of 4 -> 1 output
model = MLP(n_inputs=3, layer_sizes=[4, 4, 1])

x = [2.0, 3.0, -1.0]
out = model(x)

print("output:", out)
print("num parameters:", len(model.parameters()))