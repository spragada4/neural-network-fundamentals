import numpy as np
from rnn_numpy import RNN

# A small chunk of text -- keep it short so training is fast and the model
# can actually memorize its patterns within a reasonable number of steps
text = """the quick brown fox jumps over the lazy dog. the dog barks at the fox.
the fox runs away into the forest. the quick fox is very clever and fast."""

chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Text length: {len(text)} chars, vocab size: {vocab_size}")

hidden_size = 64
seq_length = 25
learning_rate = 0.15

rnn = RNN(vocab_size, hidden_size, seed=0)
h_prev = np.zeros((hidden_size, 1))

n_steps = 3000
pos = 0

for step in range(n_steps):
    # If we've run off the end of the text, wrap around and reset hidden state
    if pos + seq_length + 1 >= len(text):
        pos = 0
        h_prev = np.zeros((hidden_size, 1))

    input_chars = text[pos:pos + seq_length]
    target_chars = text[pos + 1:pos + seq_length + 1]  # shifted by 1: predict NEXT char

    inputs = []
    for ch in input_chars:
        x = np.zeros((vocab_size, 1))
        x[char_to_idx[ch]] = 1
        inputs.append(x)
    targets = [char_to_idx[ch] for ch in target_chars]

    ys, h_prev = rnn.forward(inputs, h_prev)
    loss = rnn.backward(targets, ys, lr=learning_rate)

    pos += seq_length

    if step % 500 == 0 or step == n_steps - 1:
        print(f"step {step:4d}  loss = {loss:.4f}")

        # Sample some text from the model to see what it's learned so far
        sample_h = h_prev.copy()
        sample_x = inputs[0]
        generated = ""
        for _ in range(80):
            sample_ys, sample_h = rnn.forward([sample_x], sample_h)
            p = np.exp(sample_ys[0] - np.max(sample_ys[0]))
            p = p / np.sum(p)
            idx = np.random.choice(vocab_size, p=p.ravel())
            generated += idx_to_char[idx]
            sample_x = np.zeros((vocab_size, 1))
            sample_x[idx] = 1
        print(f"  sample: {generated!r}\n")