import numpy as np
from rnn_numpy import RNN

np.random.seed(0)

vocab_size = 5     # pretend we have a 5-character vocabulary
hidden_size = 8

rnn = RNN(vocab_size, hidden_size, seed=0)

# Fake a 4-character sequence, one-hot encoded
seq_indices = [0, 2, 1, 3]
inputs = []
for idx in seq_indices:
    x = np.zeros((vocab_size, 1))
    x[idx] = 1
    inputs.append(x)

h0 = np.zeros((hidden_size, 1))
ys, h_final = rnn.forward(inputs, h0)

print("Number of outputs:", len(ys))          # expect 4, one per time step
print("Each output shape:", ys[0].shape)       # expect (5, 1)
print("Final hidden state shape:", h_final.shape)  # expect (8, 1)

# Now test backward runs without error and returns a loss
targets = [2, 1, 3, 0]   # pretend next-char targets
loss = rnn.backward(targets, ys, lr=0.01)
print("Loss:", loss)