# Neural Network Foundations

A hands-on guide to understanding neural networks by building them from
scratch — starting with raw scalar autograd, working up to NumPy matrix
implementations, and finally comparing against PyTorch. Every module is
runnable, tested against expected output, and documented with the *why*,
not just the *how*.

> 🚧 **Work in progress.** Built step by step, one architecture at a time.
> Currently on: LSTM (gated RNN).

## Roadmap

- [x] **01 — Foundations**: scalar autograd engine (`Value`), a `Neuron`/`Layer`/`MLP`
      built on top of it, and a full train loop with plain gradient descent
- [x] **02 — MLP**: matrix-based MLP in NumPy, trained on XOR and real MNIST (96.7%), plus a PyTorch comparison version — **complete**
- [x] **03 — CNN**: convolutions, pooling, and gradient-checked backprop, trained on MNIST, plus a PyTorch comparison — **complete**
- [ ] **04 — RNN/LSTM**: char-level RNN with BPTT working (see module README for Adagrad note) — *LSTM in progress*
- [ ] **05 — Transformer**: self-attention from scratch
- [ ] PyTorch comparison versions for each architecture above

## Why this exists

Frameworks like PyTorch hide the mechanics of training behind
`loss.backward()` and `optimizer.step()`. This repo rebuilds those
mechanics by hand first, so nothing is magic later — every layer,
every gradient, every training loop is code you can read top to bottom.
