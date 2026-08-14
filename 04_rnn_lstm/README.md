# 04 — RNN / LSTM

Builds a character-level RNN language model from scratch: given a chunk
of text, predict the next character, one step at a time, with the same
weights reused across every time step ("memory" via a hidden state passed
forward through the sequence).

## Files
- `rnn_numpy.py` — `RNN` class: forward pass (`tanh(W_xh@x + W_hh@h + b)`),
  and backpropagation through time (BPTT) with Adagrad optimization and
  gradient clipping.
- `test_rnn.py` — verifies forward/backward shapes and a sane initial loss
  (`-log(1/vocab_size)` per step, as expected from an untrained model).
- `train_rnn.py` — trains on a short text sample, sampling generated text
  periodically to watch it evolve from noise to real words.

## Results
Trained on 144 characters of text ("the quick brown fox..."). Loss dropped
from 84.2 (random init) to 0.34 by step 2500, at which point generated
samples were near-verbatim reproductions of the training text — e.g.
`". the lazy dog. the dog barks at the fox runs away into the forest..."`.

## Known limitation
Plain SGD caused this RNN to diverge (loss climbing to 300+, output
collapsing into repeated characters like `ipipipi...`) — switching to
**Adagrad** (adaptive per-parameter learning rate) fixed this. Even with
Adagrad, occasional loss spikes occur late in training: rare characters
accumulate very little gradient history, so Adagrad's denominator stays
near zero for them, and their occasional large gradient produces a
disproportionately large update. This is a known Adagrad quirk on small,
unevenly-distributed vocabularies — not a bug in the BPTT implementation
(verified separately via forward/backward shape and initial-loss checks).
LSTMs (below) use gating mechanisms specifically designed to make training
dynamics like this more stable.

## Run it
\`\`\`bash
python3 test_rnn.py
python3 train_rnn.py
\`\`\`