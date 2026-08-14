import numpy as np
from lstm_numpy import LSTM

np.random.seed(0)
vocab_size, hidden_size = 5, 8
lstm = LSTM(vocab_size, hidden_size, seed=0)

seq_indices = [0, 2, 1, 3]
inputs = []
for idx in seq_indices:
    x = np.zeros((vocab_size, 1))
    x[idx] = 1
    inputs.append(x)

h0 = np.zeros((hidden_size, 1))
c0 = np.zeros((hidden_size, 1))
ys, h_final, c_final = lstm.forward(inputs, h0, c0)

print("Number of outputs:", len(ys))
print("Each output shape:", ys[0].shape)
print("Final h shape:", h_final.shape)
print("Final c shape:", c_final.shape)

targets = [2, 1, 3, 0]
loss = lstm.backward(targets, ys, lr=0.01)
print("Loss:", loss)   # expect ~ -log(1/5)*4 ~= 6.4, like the RNN test