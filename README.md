# Neural Network Foundations

A hands-on guide to understanding neural networks by building them from
scratch — starting with raw scalar autograd, working up to NumPy matrix
implementations, and finally comparing against PyTorch. Every module is
runnable, tested against expected output, and documented with the *why*,
not just the *how*.

> 🚧 **Work in progress.** Built step by step, one architecture at a time.
> Currently on: matrix-based MLP (NumPy).

## Roadmap

- [x] **01 — Foundations**: scalar autograd engine (`Value`), a `Neuron`/`Layer`/`MLP`
      built on top of it, and a full train loop with plain gradient descent
- [ ] **02 — MLP**: matrix-based MLP in NumPy (batched forward/backward pass,
      softmax + cross-entropy, trained on XOR) — *in progress*
- [ ] **03 — CNN**: convolutions, pooling, and image classification from scratch
- [ ] **04 — RNN/LSTM**: sequence modeling from scratch
- [ ] **05 — Transformer**: self-attention from scratch
- [ ] PyTorch comparison versions for each architecture above

## Why this exists

Frameworks like PyTorch hide the mechanics of training behind
`loss.backward()` and `optimizer.step()`. This repo rebuilds those
mechanics by hand first, so nothing is magic later — every layer,
every gradient, every training loop is code you can read top to bottom.
