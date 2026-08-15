import torch
import torch.nn as nn
import numpy as np
from transformer_pytorch import CharTransformer, causal_mask

torch.manual_seed(0)

text = """the quick brown fox jumps over the lazy dog. the dog barks at the fox.
the fox runs away into the forest. the quick fox is very clever and fast."""

chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

seq_length = 25
model = CharTransformer(vocab_size, d_model=64, n_heads=4, d_ff=128, n_layers=2, max_len=seq_length + 1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # Adam: the de facto standard for transformers

mask = causal_mask(seq_length)

n_steps = 2000
pos = 0

for step in range(n_steps):
    if pos + seq_length + 1 >= len(text):
        pos = 0

    input_chars = text[pos:pos + seq_length]
    target_chars = text[pos + 1:pos + seq_length + 1]

    x = torch.tensor([[char_to_idx[ch] for ch in input_chars]], dtype=torch.long)   # (1, seq_len)
    y = torch.tensor([char_to_idx[ch] for ch in target_chars], dtype=torch.long)     # (seq_len,)

    optimizer.zero_grad()
    logits = model(x, mask=mask)          # (1, seq_len, vocab_size)
    loss = criterion(logits.squeeze(0), y)
    loss.backward()
    optimizer.step()

    pos += seq_length

    if step % 400 == 0 or step == n_steps - 1:
        print(f"step {step:4d}  loss = {loss.item():.4f}")

        with torch.no_grad():
            seed_chars = input_chars[:5]
            generated = seed_chars
            for _ in range(80):
                context = generated[-seq_length:]
                xin = torch.tensor([[char_to_idx[ch] for ch in context]], dtype=torch.long)
                m = causal_mask(len(context))
                out = model(xin, mask=m)
                p = torch.softmax(out[0, -1], dim=0).numpy()
                idx = np.random.choice(vocab_size, p=p)
                generated += idx_to_char[idx]
        print(f"  sample: {generated!r}\n")