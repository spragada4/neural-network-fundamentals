import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q, K, V: (seq_len, d_k) for a single sequence (no batch dim yet).
    mask: optional (seq_len, seq_len) array of 0s and -inf, added before
          softmax -- used later to stop positions from attending to future
          tokens (needed for language modeling).
    Returns: (output, attention_weights)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)      # (seq_len, seq_len): every query vs every key

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores, axis=-1)   # each row sums to 1 -- a distribution over positions
    output = weights @ V                 # weighted blend of values
    return output, weights