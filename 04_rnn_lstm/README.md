# 04 — RNN / LSTM

Character-level language models: given a chunk of text, predict the next
character, one step at a time. Demonstrates the core weakness of plain
RNNs (unstable long-range gradients) and how LSTM's gating mechanism
fixes it — reproduced empirically, not just described.

## Files
- `rnn_numpy.py` — plain `RNN`: `tanh(W_xh@x + W_hh@h + b)`, with
  backpropagation through time (BPTT) and Adagrad.
- `test_rnn.py` — shape/baseline-loss verification.
- `train_rnn.py` — trains the RNN on a short text sample.
- `lstm_numpy.py` — `LSTM`: forget/input/output gates plus a cell state
  updated additively (`c_t = f_t * c_{t-1} + i_t * c̃_t`), with full BPTT
  through all four gates and Adagrad.
- `test_lstm.py` — shape/baseline-loss verification.
- `train_lstm.py` — trains the LSTM on the identical text/hyperparameters
  as the RNN, for direct comparison.
- `rnn_lstm_pytorch.py` — `nn.LSTM` version for comparison.

## Results (144-char text, 3000 steps, lr=0.15, Adagrad)
| Version | Final loss | Late-training spike? |
|---|---|---|
| Plain RNN (from scratch) | 0.0137 (best: step 2500), spiked to 42.0 at step 2999 | **Yes** |
| LSTM (from scratch) | 0.0137 (steadily, no spike) | No |
| LSTM (PyTorch) | 0.0006 | No |

## The key lesson
The plain RNN reuses the same `W_hh` matrix at every time step, so
gradients flowing backward through many steps get multiplied by the same
matrix (and squashed through `tanh`) repeatedly — this can explode or
vanish, and combined with Adagrad's near-zero denominator for rare
characters, produced a real destabilization late in training (see
`train_rnn.py` output: loss 0.34 → 42.0 in the last few hundred steps).

The LSTM's cell state update is **additive**
(`c_t = f_t * c_{t-1} + i_t * c̃_t`) and gated by a *learned, bounded*
forget gate `f_t ∈ (0,1)`, rather than a repeated raw matrix multiply.
Gradient flowing back through the cell state only ever gets scaled by
`f_t`, which the network can learn to keep close to 1 when it needs to
preserve information over many steps — this is *why* LSTMs largely solved
the vanishing/exploding gradient problem that limited plain RNNs, and it's
visible directly in the loss curves above, not just in theory.

## Run it
\`\`\`bash
python3 test_rnn.py && python3 train_rnn.py
python3 test_lstm.py && python3 train_lstm.py
python3 rnn_lstm_pytorch.py
\`\`\`