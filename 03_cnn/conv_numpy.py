import numpy as np


def conv2d_single(image, kernel, stride=1):
    """Slide `kernel` over a single 2D `image`, no padding.
    image:  (H, W)
    kernel: (kH, kW)
    returns: (out_H, out_W)
    """
    H, W = image.shape
    kH, kW = kernel.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1

    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            row, col = i * stride, j * stride
            patch = image[row:row + kH, col:col + kW]
            out[i, j] = np.sum(patch * kernel)   # elementwise multiply + sum = dot product
    return out