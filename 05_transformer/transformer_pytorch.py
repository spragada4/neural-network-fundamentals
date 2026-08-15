import torch
import torch.nn as nn
import math

torch.manual_seed(0)


class PositionalEncoding(nn.Module):
    """Attention has no built-in sense of order (unlike RNN's sequential
    processing or CNN's local windows) -- it looks at all positions at
    once, symmetrically. So we inject position information directly into
    the input embeddings using fixed sine/cosine waves of different
    frequencies, one set per dimension."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.shape[1], :]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        # batch_first=True: shapes are (batch, seq_len, d_model), matching
        # the (seq_len, d_model) convention from your NumPy version, plus a batch dim
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Residual connection: add the input BACK to attention's output,
        # so gradients have a direct path around the attention block --
        # this is what makes very deep transformers trainable at all
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class CharTransformer(nn.Module):
    """Same char-level next-character prediction task as the RNN/LSTM,
    now solved with self-attention instead of recurrence."""

    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=128, n_layers=2, max_len=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pos_encode(x)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.fc_out(x)


def causal_mask(seq_len):
    """Prevents position i from attending to positions > i -- essential for
    language modeling, where predicting position i must not 'cheat' by
    looking at future characters it's supposed to predict."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))