import numpy as np


class RNN:
    def __init__(self, vocab_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Small random init, scaled down -- RNN weights are more prone to
        # exploding than MLP/CNN weights since the same matrix multiplies
        # the hidden state over and over across time steps
        self.W_xh = rng.normal(0, 0.01, size=(hidden_size, vocab_size))
        self.W_hh = rng.normal(0, 0.01, size=(hidden_size, hidden_size))
        self.W_hy = rng.normal(0, 0.01, size=(vocab_size, hidden_size))
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((vocab_size, 1))

    def forward(self, inputs, h_prev):
        """inputs: list of one-hot column vectors (vocab_size, 1), one per time step.
        h_prev: (hidden_size, 1) initial hidden state.
        Returns: list of hidden states, list of output logits, final hidden state.
        """
        hs, ys = [], []
        h = h_prev
        self.inputs = inputs  # cache for backward
        self.hs = [h_prev]    # hs[t] = hidden state AFTER processing inputs[t]

        for x in inputs:
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
            y = self.W_hy @ h + self.b_y   # raw logits, softmax applied outside
            self.hs.append(h)
            ys.append(y)

        return ys, h

    def backward(self, targets, ys, lr):
        """targets: list of integer char indices (the correct next char at each step).
        ys: list of logits from forward().
        Implements backpropagation through time (BPTT): gradients flow backward
        through every time step, accumulating into the SAME shared weight matrices.
        """
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros((self.hidden_size, 1))  # gradient from the FUTURE time step

        loss = 0.0
        T = len(self.inputs)

        for t in reversed(range(T)):
            # softmax + cross-entropy gradient, same (pred - true) trick as MLP/CNN
            p = np.exp(ys[t] - np.max(ys[t])) / np.sum(np.exp(ys[t] - np.max(ys[t])))
            loss += -np.log(p[targets[t], 0] + 1e-9)

            dy = p.copy()
            dy[targets[t]] -= 1   # (pred - true)

            dW_hy += dy @ self.hs[t + 1].T
            db_y += dy

            # gradient flows into hidden state from TWO places: this step's
            # output, AND the next time step's hidden state (since h_t feeds h_(t+1))
            dh = self.W_hy.T @ dy + dh_next
            dh_raw = (1 - self.hs[t + 1] ** 2) * dh   # tanh derivative: 1 - tanh(x)^2

            db_h += dh_raw
            dW_xh += dh_raw @ self.inputs[t].T
            dW_hh += dh_raw @ self.hs[t].T

            dh_next = self.W_hh.T @ dh_raw   # pass gradient back to the PREVIOUS time step

        # Gradient clipping: RNN gradients can explode across many time steps,
        # so clip each gradient to a sane range before updating
        for grad in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(grad, -5, 5, out=grad)

        self.W_xh -= lr * dW_xh
        self.W_hh -= lr * dW_hh
        self.W_hy -= lr * dW_hy
        self.b_h -= lr * db_h
        self.b_y -= lr * db_y

        return loss