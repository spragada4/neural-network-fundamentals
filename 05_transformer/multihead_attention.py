import numpy as np
from attention_numpy import softmax


class MultiHeadAttention:
    def __init__(self, d_model, n_heads, seed=0):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # dimension per head

        scale = np.sqrt(2.0 / d_model)
        self.W_q = rng.normal(0, scale, size=(d_model, d_model))
        self.W_k = rng.normal(0, scale, size=(d_model, d_model))
        self.W_v = rng.normal(0, scale, size=(d_model, d_model))
        self.W_o = rng.normal(0, scale, size=(d_model, d_model))   # output projection

    def split_heads(self, X):
        """X: (seq_len, d_model) -> (n_heads, seq_len, d_k)"""
        seq_len = X.shape[0]
        X = X.reshape(seq_len, self.n_heads, self.d_k)
        return X.transpose(1, 0, 2)

    def combine_heads(self, X):
        """X: (n_heads, seq_len, d_k) -> (seq_len, d_model)"""
        n_heads, seq_len, d_k = X.shape
        X = X.transpose(1, 0, 2)
        return X.reshape(seq_len, n_heads * d_k)

    def forward(self, X, mask=None):
        """X: (seq_len, d_model). Single sequence, no batch dim, for clarity."""
        self.X = X
        self.Q = X @ self.W_q
        self.K = X @ self.W_k
        self.V = X @ self.W_v

        Qh = self.split_heads(self.Q)   # (n_heads, seq_len, d_k)
        Kh = self.split_heads(self.K)
        Vh = self.split_heads(self.V)

        self.head_outputs = []
        self.head_weights = []
        for h in range(self.n_heads):
            scores = Qh[h] @ Kh[h].T / np.sqrt(self.d_k)
            if mask is not None:
                scores = scores + mask
            weights = softmax(scores, axis=-1)
            out = weights @ Vh[h]
            self.head_outputs.append(out)
            self.head_weights.append(weights)

        concat = self.combine_heads(np.stack(self.head_outputs))  # (seq_len, d_model)
        self.concat = concat
        output = concat @ self.W_o
        return output