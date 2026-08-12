# 03 — CNN (Convolutional Neural Network)

Builds a CNN from raw NumPy: a hand-written 2D convolution, a trainable
`Conv2D` layer (via the im2col trick), a `MaxPool2D` layer, and a full
Conv→ReLU→Pool→Dense model trained end-to-end on MNIST images (not
flattened vectors — actual 2D image tensors).

## Files
- `conv_numpy.py` — the simplest possible 2D convolution (nested loops),
  built to make the sliding-window operation visible and verifiable by hand.
- `conv_layer.py` — a real trainable `Conv2D` layer: multi-channel,
  multi-filter, with a full backward pass using the im2col trick
  (reshaping patches so convolution becomes one matrix multiply).
- `pool_layer.py` — `MaxPool2D`, forward and backward. Only the position
  that was the max in each window receives gradient; everything else gets 0.
- `cnn_model.py` — `SimpleCNN`: Conv → ReLU → MaxPool → Flatten → Dense → Softmax.
- `train_cnn.py` — trains the from-scratch CNN on a 5,000-image MNIST subset.
- `cnn_pytorch.py` — the same architecture in PyTorch (`nn.Conv2d`,
  `nn.MaxPool2d`, `nn.Linear`), trained on the identical subset for comparison.
- `test_conv.py`, `test_conv_layer.py`, `test_pool.py` — verification tests,
  including a **numerical gradient check**: perturbing a weight by a tiny
  amount and confirming the resulting loss change matches the analytically
  computed gradient exactly. This is a standard technique for validating
  any from-scratch backward pass.

## Results (5,000 train images, 1,000 test images, 3 epochs)
| Version | Test accuracy |
|---|---|
| NumPy (from scratch) | 86.9% |
| PyTorch | 85.8% |

Lower than the MLP's 96.7% — expected, since this trains on ~8% of the
data for fewer epochs, specifically to keep the loop-based backward pass
fast enough to iterate on. Not a like-for-like comparison with the MLP.

## Key mapping: from-scratch → PyTorch
| From scratch | PyTorch |
|---|---|
| `conv_numpy.py` (loop-based) | conceptual reference only |
| `Conv2D` (im2col) | `nn.Conv2d` |
| `MaxPool2D` | `nn.MaxPool2d` |
| manual chain-rule backward (dense→pool→relu→conv) | `loss.backward()` |

## Known limitation / stretch goal
`MaxPool2D.backward` uses nested Python `for` loops over batch and channel
dims, which is slow. PyTorch's built-in layers are vectorized in C++/CUDA,
which is why `cnn_pytorch.py` runs dramatically faster on identical data.
**Optional improvement:** vectorize `MaxPool2D.backward` using NumPy
indexing (e.g. `np.add.at`) to remove the Python loops, enabling training
on the full 60k-image MNIST dataset in reasonable time.

## Run it
\`\`\`bash
python3 test_conv.py
python3 test_conv_layer.py
python3 test_pool.py
python3 train_cnn.py       # from-scratch CNN
python3 cnn_pytorch.py     # PyTorch comparison
\`\`\`