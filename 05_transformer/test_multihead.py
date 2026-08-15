import numpy as np
from multihead_attention import MultiHeadAttention

np.random.seed(0)

seq_len, d_model, n_heads = 4, 8, 2
mha = MultiHeadAttention(d_model, n_heads, seed=0)

X = np.random.randn(seq_len, d_model)
output = mha.forward(X)

print("Input shape: ", X.shape)
print("Output shape:", output.shape)   # should match input: (4, 8)

# Verify split_heads -> combine_heads is a perfect round trip (no data lost/reordered)
test = np.random.randn(seq_len, d_model)
split = mha.split_heads(test)
print("\nSplit shape:", split.shape)    # (n_heads=2, seq_len=4, d_k=4)
recombined = mha.combine_heads(split)
print("Round-trip matches original:", np.allclose(test, recombined))

print("\nNumber of attention heads computed:", len(mha.head_weights))
print("Each head's attention weights shape:", mha.head_weights[0].shape)