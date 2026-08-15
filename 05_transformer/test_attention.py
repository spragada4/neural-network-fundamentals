import numpy as np
from attention_numpy import scaled_dot_product_attention

# 3 positions, d_k = 2. Make position 0's query nearly IDENTICAL to
# position 1's key, and very different from positions 0 and 2's keys --
# so position 0 should attend almost entirely to position 1.
Q = np.array([
    [1.0, 0.0],   # query 0
    [0.0, 1.0],   # query 1
    [1.0, 1.0],   # query 2
])
K = np.array([
    [0.0, 1.0],   # key 0 -- far from query 0
    [1.0, 0.0],   # key 1 -- matches query 0 closely!
    [0.5, 0.5],   # key 2
])
V = np.array([
    [10.0, 0.0],   # value 0
    [0.0, 10.0],   # value 1 -- this is what query 0 should mostly retrieve
    [5.0, 5.0],    # value 2
])

output, weights = scaled_dot_product_attention(Q, K, V)

print("Attention weights (each row sums to 1):\n", np.round(weights, 3))
print("\nOutput:\n", np.round(output, 3))
print("\nRow sums (sanity check, should all be 1.0):", weights.sum(axis=1))