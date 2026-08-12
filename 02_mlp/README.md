# 02 — MLP (Matrix-Based)

Moves from the scalar `Value` engine in `01_foundations` to real matrix
operations — the way production neural networks are actually implemented.
Whole layers are computed as one matrix multiply instead of looping over
individual neurons.

## Files
- `mlp_numpy.py` — `MLP` class: forward pass (`X @ W + b`, ReLU, softmax),
  manual backward pass (batched gradients), and cross-entropy loss.
- `mnist_loader.py` — downloads and parses the MNIST dataset (shared by
  both the NumPy and PyTorch versions).
- `test_mlp.py` — trains on XOR (not linearly separable — proves the
  hidden layer + nonlinearity is doing real work).
- `train_mnist.py` — trains the NumPy MLP on real MNIST digit images.
- `mlp_pytorch.py` — the same architecture in PyTorch (`nn.Linear`,
  `CrossEntropyLoss`, `optim.SGD`), to compare against the from-scratch
  version line by line.

## Results
| Version | Test accuracy (5 epochs) |
|---|---|
| NumPy (from scratch) | ~96.7% |
| PyTorch | ~96.5% |

The two land in the same range, confirming the from-scratch math matches
what PyTorch does internally — small differences are just random init /
implementation noise, not a correctness gap.

## Key mapping: from-scratch → PyTorch
| From scratch | PyTorch |
|---|---|
| `W @ x + b` | `nn.Linear` |
| manual softmax + cross-entropy backward | `nn.CrossEntropyLoss` (expects raw logits) |
| `p.grad = 0.0` reset | `optimizer.zero_grad()` |
| manual chain-rule backward pass | `loss.backward()` |
| `W -= lr * dW` | `optimizer.step()` |

## Run it
\`\`\`bash
python3 test_mlp.py        # XOR sanity check
python3 train_mnist.py     # NumPy version on real MNIST
python3 mlp_pytorch.py     # PyTorch version on real MNIST
\`\`\`