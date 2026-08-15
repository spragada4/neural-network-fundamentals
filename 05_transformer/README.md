# 05 — Transformer (Self-Attention)

Implements scaled dot-product attention and multi-head attention from
scratch in NumPy (forward pass only, verified against hand-designed
examples), then assembles a full Transformer block and trains a
char-level language model in PyTorch — the way transformers are actually
trained in practice.

## Files
- `attention_numpy.py` — `scaled_dot_product_attention`: `softmax(QK^T / sqrt(d_k)) @ V`.
- `test_attention.py` — verifies attention weights against a hand-designed
  case where the correct answer (which position should be attended to
  most) is known in advance.
- `multihead_attention.py` — `MultiHeadAttention`: splits Q/K/V into
  multiple heads, runs attention per head, recombines. Forward-only —
  see note below on why backward is left to PyTorch.
- `test_multihead.py` — verifies shapes and that splitting/recombining
  heads is a lossless round trip.
- `transformer_pytorch.py` — full encoder block (attention + residual +
  layer norm + feedforward + residual + layer norm), positional encoding
  (sine/cosine, since attention has no inherent sense of order), and a
  causal mask (blocks attending to future positions — required for
  language modeling).
- `train_transformer.py` — trains a 2-layer, 4-head char-level transformer
  on the same text and task as the RNN/LSTM, for comparison.

## Why no from-scratch backward pass
Every previous module in this repo (MLP, CNN, RNN, LSTM) included a full
hand-written backward pass, verified with gradient checking. The
Transformer's backward pass involves gradients through multiple attention
heads, residual connections, and layer norm simultaneously — the
bookkeeping is very large without teaching new *concepts* beyond what
BPTT (Step 13-16) already covered: it's still the same chain rule, just
with more terms. In practice, nobody hand-derives transformer backprop —
even at Anthropic, PyTorch's autograd handles it. This module verifies
the **forward math** by hand (the part that defines what a transformer
actually computes), then hands training off to PyTorch, matching real-world practice.

## Results (144-char text, 2000 steps)
| Version | Final training loss | Generation quality |
|---|---|---|
| LSTM | 0.0006 | Clean, near-verbatim reproduction of training text |
| Transformer | 0.0005 | Degrades into repetitive loops (e.g. `fororororo`) despite lower loss |

## Key lesson: low loss ≠ good generation
This is a real and important phenomenon, not a bug. Training uses
**teacher forcing** — the model always sees the true previous characters
as context, never its own predictions. On a tiny 144-character dataset
with a comparatively large model, this lets loss collapse toward zero
via near-memorization. But **generation is autoregressive** — the model
feeds its own sampled output back in as context. Once a sampled character
drifts slightly from the training distribution (likely once generation
runs past the 144 characters the model has memorized), the model has
never seen that scenario during training and has no learned way to
recover, often collapsing into a repetitive loop. This train/generation
mismatch is a well-documented issue (sometimes called exposure bias) and
is exactly why **always inspecting actual generated output, not just the
loss curve, matters** — a loss number alone can be misleading.

## Run it
\`\`\`bash
python3 test_attention.py
python3 test_multihead.py
python3 train_transformer.py
\`\`\`