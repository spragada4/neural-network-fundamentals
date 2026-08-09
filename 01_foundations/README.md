# 01 — Foundations

Before building CNNs or RNNs, you need the one idea underneath all of them:
a neural network is a math expression, and training it means computing
gradients of that expression via the chain rule (backpropagation).

## Files
- `value.py` — a scalar autograd engine (~60 lines). Every operation
  (`+`, `*`, `**`, `relu`) records how to pass gradient back to its inputs.
  `.backward()` walks the computation graph in reverse topological order
  and accumulates gradients everywhere.
- `nn.py` — `Neuron`, `Layer`, `MLP` built entirely out of `Value`.
- `test_value.py` — verifies gradients against hand-computed calculus.
- `test_nn.py` — verifies the forward pass and parameter count.
- `train_toy.py` — trains a tiny MLP on 4 toy examples using plain
  gradient descent (no optimizer library).

## Key lesson learned
Only apply nonlinearities (ReLU, etc.) on **hidden** layers. Applying
ReLU on the **output** layer can zero out the pre-activation and kill
gradient flow permanently ("dead ReLU") if the target requires negative
values.

## Run it
\`\`\`bash
python3 test_value.py
python3 test_nn.py
python3 train_toy.py
\`\`\`