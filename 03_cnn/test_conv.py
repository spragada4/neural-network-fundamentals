import numpy as np
from conv_numpy import conv2d_single

# A simple 4x4 "image" with a vertical edge (left half bright, right half dark)
image = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 0, 0],
], dtype=float)

# A vertical edge detector: positive on the left, negative on the right
kernel = np.array([
    [1, -1],
    [1, -1],
], dtype=float)

out = conv2d_single(image, kernel)
print("Output:\n", out)