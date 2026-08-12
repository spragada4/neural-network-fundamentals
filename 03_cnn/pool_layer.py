import numpy as np


class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, X):
        """X: (N, C, H, W) -> out: (N, C, out_H, out_W)"""
        self.X = X
        N, C, H, W = X.shape
        s, st = self.size, self.stride
        out_H = (H - s) // st + 1
        out_W = (W - s) // st + 1

        out = np.zeros((N, C, out_H, out_W))
        # remember WHERE the max came from, so backward knows where to send gradient
        self.max_idx = {}

        for i in range(out_H):
            for j in range(out_W):
                row, col = i * st, j * st
                window = X[:, :, row:row + s, col:col + s]        # (N, C, s, s)
                window_flat = window.reshape(N, C, -1)
                out[:, :, i, j] = np.max(window_flat, axis=2)
                self.max_idx[(i, j)] = np.argmax(window_flat, axis=2)  # (N, C)

        return out

    def backward(self, dout):
        """dout: (N, C, out_H, out_W) -> dX: same shape as input"""
        N, C, H, W = self.X.shape
        s, st = self.size, self.stride
        out_H, out_W = dout.shape[2], dout.shape[3]

        dX = np.zeros_like(self.X)
        for i in range(out_H):
            for j in range(out_W):
                row, col = i * st, j * st
                idx = self.max_idx[(i, j)]           # (N, C) -- flat index of the max within each window
                idx_row = idx // s
                idx_col = idx % s

                # only the position that WAS the max gets any gradient --
                # everything else in the window contributed nothing to the output
                for n in range(N):
                    for c in range(C):
                        dX[n, c, row + idx_row[n, c], col + idx_col[n, c]] += dout[n, c, i, j]
        return dX