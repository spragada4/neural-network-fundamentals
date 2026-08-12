import numpy as np


def get_patches(X, kH, kW, stride=1):
    """Extract all sliding-window patches from a batch of images, im2col style.
    X: (N, C, H, W)
    returns: patches of shape (N, out_H, out_W, C, kH, kW)
    """
    N, C, H, W = X.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1

    # Use stride tricks to build a view over all patches without copying data
    s0, s1, s2, s3 = X.strides
    shape = (N, out_H, out_W, C, kH, kW)
    strides = (s0, s2 * stride, s3 * stride, s1, s2, s3)
    patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    return patches, out_H, out_W


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, seed=0):
        rng = np.random.default_rng(seed)
        self.stride = stride
        self.kH = self.kW = kernel_size
        self.C_in = in_channels
        self.C_out = out_channels

        # He init, same idea as the MLP -- keeps activation scale stable
        fan_in = in_channels * kernel_size * kernel_size
        self.W = rng.normal(0, np.sqrt(2.0 / fan_in),
                             size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros(out_channels)

    def forward(self, X):
        """X: (N, C_in, H, W) -> out: (N, C_out, out_H, out_W)"""
        self.X = X
        N = X.shape[0]
        patches, out_H, out_W = get_patches(X, self.kH, self.kW, self.stride)
        self.patches = patches
        self.out_H, self.out_W = out_H, out_W

        # Flatten patches to (N*out_H*out_W, C_in*kH*kW) and filters to (C_in*kH*kW, C_out)
        patches_flat = patches.reshape(N * out_H * out_W, -1)
        W_flat = self.W.reshape(self.C_out, -1).T   # (C_in*kH*kW, C_out)

        out = patches_flat @ W_flat + self.b         # (N*out_H*out_W, C_out)
        out = out.reshape(N, out_H, out_W, self.C_out)
        out = out.transpose(0, 3, 1, 2)               # -> (N, C_out, out_H, out_W)
        return out

    def backward(self, dout, lr):
        """dout: (N, C_out, out_H, out_W), gradient of loss w.r.t. this layer's output."""
        N = dout.shape[0]
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, self.C_out)  # (N*out_H*out_W, C_out)

        patches_flat = self.patches.reshape(N * self.out_H * self.out_W, -1)

        # Gradient w.r.t. filters: how much each patch contributed to each output channel
        dW_flat = patches_flat.T @ dout_flat            # (C_in*kH*kW, C_out)
        dW = dW_flat.T.reshape(self.W.shape)
        db = dout_flat.sum(axis=0)

        # Gradient w.r.t. input: scatter-add each output gradient back over the
        # input positions its patch came from
        W_flat = self.W.reshape(self.C_out, -1)          # (C_out, C_in*kH*kW)
        dpatches_flat = dout_flat @ W_flat                # (N*out_H*out_W, C_in*kH*kW)
        dpatches = dpatches_flat.reshape(N, self.out_H, self.out_W, self.C_in, self.kH, self.kW)

        dX = np.zeros_like(self.X)
        for i in range(self.out_H):
            for j in range(self.out_W):
                row, col = i * self.stride, j * self.stride
                dX[:, :, row:row + self.kH, col:col + self.kW] += dpatches[:, i, j]

        self.W -= lr * dW
        self.b -= lr * db
        return dX