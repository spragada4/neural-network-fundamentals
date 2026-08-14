import torch
import torch.nn as nn
import numpy as np

text = """the quick brown fox jumps over the lazy dog. the dog barks at the fox.
the fox runs away into the forest. the quick fox is very clever and fast."""

chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

hidden_size = 64
seq_length = 25


class CharLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # batch_first=False: input shape (seq_len, batch, vocab_size), matching
        # how you fed one-hot vectors one time step at a time in lstm_numpy.py
        self.lstm = nn.LSTM(vocab_size, hidden_size)   # replaces your 4 gates
        self.fc = nn.Linear(hidden_size, vocab_size)   # replaces your W_hy, b_y

    def forward(self, x, h_c=None):
        out, h_c = self.lstm(x, h_c)   # out: (seq_len, batch, hidden_size)
        logits = self.fc(out)
        return logits, h_c


model = CharLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.15)  # same optimizer as your version

n_steps = 3000
pos = 0
h_c = None

for step in range(n_steps):
    if pos + seq_length + 1 >= len(text):
        pos = 0
        h_c = None   # reset hidden/cell state, same as your h_prev/c_prev reset

    input_chars = text[pos:pos + seq_length]
    target_chars = text[pos + 1:pos + seq_length + 1]

    x = torch.zeros(seq_length, 1, vocab_size)
    for t, ch in enumerate(input_chars):
        x[t, 0, char_to_idx[ch]] = 1
    targets = torch.tensor([char_to_idx[ch] for ch in target_chars], dtype=torch.long)

    if h_c is not None:
        h_c = (h_c[0].detach(), h_c[1].detach())   # detach so gradients don't flow across resets

    optimizer.zero_grad()
    logits, h_c = model(x, h_c)
    loss = criterion(logits.squeeze(1), targets)
    loss.backward()
    optimizer.step()

    pos += seq_length

    if step % 500 == 0 or step == n_steps - 1:
        print(f"step {step:4d}  loss = {loss.item():.4f}")

        with torch.no_grad():
            sample_h_c = h_c
            sample_x = x[0:1]
            generated = ""
            for _ in range(80):
                sample_logits, sample_h_c = model(sample_x, sample_h_c)
                p = torch.softmax(sample_logits[0, 0], dim=0).numpy()
                idx = np.random.choice(vocab_size, p=p)
                generated += idx_to_char[idx]
                sample_x = torch.zeros(1, 1, vocab_size)
                sample_x[0, 0, idx] = 1
        print(f"  sample: {generated!r}\n")