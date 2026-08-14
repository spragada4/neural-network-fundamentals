import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTM:
    def __init__(self, vocab_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        z_size = hidden_size + vocab_size   # concat of [h_prev, x]

        # One combined weight matrix per gate, each mapping the concatenated
        # [h_prev, x] vector to hidden_size outputs
        def init_gate():
            return rng.normal(0, 0.01, size=(hidden_size, z_size))

        self.W_f, self.b_f = init_gate(), np.zeros((hidden_size, 1))
        self.W_i, self.b_i = init_gate(), np.zeros((hidden_size, 1))
        self.W_c, self.b_c = init_gate(), np.zeros((hidden_size, 1))
        self.W_o, self.b_o = init_gate(), np.zeros((hidden_size, 1))
        self.W_hy = rng.normal(0, 0.01, size=(vocab_size, hidden_size))
        self.b_y = np.zeros((vocab_size, 1))

        # Adagrad memory for every parameter
        self.mem = {name: np.zeros_like(getattr(self, name)) for name in
                    ["W_f", "b_f", "W_i", "b_i", "W_c", "b_c", "W_o", "b_o", "W_hy", "b_y"]}

    def forward(self, inputs, h_prev, c_prev):
        """inputs: list of one-hot vectors. Returns outputs, final h, final c."""
        self.inputs = inputs
        T = len(inputs)
        # cache everything needed for backward, indexed by time step
        self.z, self.f, self.i, self.c_tilde, self.c, self.o, self.h = (
            [None] * T, [None] * T, [None] * T, [None] * T, [None] * T, [None] * T, [None] * T
        )
        self.c_prev_cache = [None] * T
        ys = []

        h, c = h_prev, c_prev
        for t, x in enumerate(inputs):
            z = np.vstack([h, x])                      # concat [h_prev, x]
            self.z[t] = z
            self.c_prev_cache[t] = c

            f = sigmoid(self.W_f @ z + self.b_f)
            i = sigmoid(self.W_i @ z + self.b_i)
            c_tilde = np.tanh(self.W_c @ z + self.b_c)
            c = f * c + i * c_tilde                      # additive cell update
            o = sigmoid(self.W_o @ z + self.b_o)
            h = o * np.tanh(c)

            self.f[t], self.i[t], self.c_tilde[t] = f, i, c_tilde
            self.c[t], self.o[t], self.h[t] = c, o, h

            y = self.W_hy @ h + self.b_y
            ys.append(y)

        return ys, h, c

    def backward(self, targets, ys, lr):
        T = len(self.inputs)
        grads = {name: np.zeros_like(getattr(self, name)) for name in self.mem}
        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))
        loss = 0.0

        for t in reversed(range(T)):
            p = np.exp(ys[t] - np.max(ys[t])) / np.sum(np.exp(ys[t] - np.max(ys[t])))
            loss += -np.log(p[targets[t], 0] + 1e-9)
            dy = p.copy()
            dy[targets[t]] -= 1

            grads["W_hy"] += dy @ self.h[t].T
            grads["b_y"] += dy

            dh = self.W_hy.T @ dy + dh_next
            do = dh * np.tanh(self.c[t])
            do_raw = do * self.o[t] * (1 - self.o[t])          # sigmoid derivative

            dc = dh * self.o[t] * (1 - np.tanh(self.c[t]) ** 2) + dc_next
            df = dc * self.c_prev_cache[t]
            df_raw = df * self.f[t] * (1 - self.f[t])

            di = dc * self.c_tilde[t]
            di_raw = di * self.i[t] * (1 - self.i[t])

            dc_tilde = dc * self.i[t]
            dc_tilde_raw = dc_tilde * (1 - self.c_tilde[t] ** 2)  # tanh derivative

            z = self.z[t]
            grads["W_f"] += df_raw @ z.T
            grads["b_f"] += df_raw
            grads["W_i"] += di_raw @ z.T
            grads["b_i"] += di_raw
            grads["W_c"] += dc_tilde_raw @ z.T
            grads["b_c"] += dc_tilde_raw
            grads["W_o"] += do_raw @ z.T
            grads["b_o"] += do_raw

            dz = (self.W_f.T @ df_raw + self.W_i.T @ di_raw +
                  self.W_c.T @ dc_tilde_raw + self.W_o.T @ do_raw)
            dh_next = dz[:self.hidden_size, :]     # first part of z was h_prev
            dc_next = dc * self.f[t]               # cell gradient carries through the forget gate

        for name in grads:
            np.clip(grads[name], -5, 5, out=grads[name])
            self.mem[name] += grads[name] ** 2
            param = getattr(self, name)
            param -= lr * grads[name] / (np.sqrt(self.mem[name]) + 1e-8)

        return loss