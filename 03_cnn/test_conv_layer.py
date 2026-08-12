import numpy as np
from conv_layer import Conv2D

np.random.seed(0)

# Batch of 2 images, 3 channels (like RGB), 8x8 pixels
X = np.random.randn(2, 3, 8, 8)

conv = Conv2D(in_channels=3, out_channels=4, kernel_size=3, stride=1, seed=0)
out = conv.forward(X)
print("Output shape:", out.shape)   # expect (2, 4, 6, 6): 8-3+1=6

# Fake upstream gradient (as if this fed into a loss)
dout = np.random.randn(*out.shape)
dX = conv.backward(dout, lr=0.01)
print("dX shape:", dX.shape)        # should match X.shape exactly: (2, 3, 8, 8)

# --- Numerical gradient check on a single weight ---
# Perturb one weight slightly, see if the change in output matches our computed gradient
conv2 = Conv2D(in_channels=3, out_channels=4, kernel_size=3, stride=1, seed=0)
eps = 1e-5
i0, i1, i2, i3 = 0, 0, 0, 0   # which weight to test

out1 = conv2.forward(X).copy()
loss1 = np.sum(out1 * dout)   # pretend loss = sum(output * dout), so d(loss)/d(out) = dout

conv2.W[i0, i1, i2, i3] += eps
out2 = conv2.forward(X)
loss2 = np.sum(out2 * dout)

numerical_grad = (loss2 - loss1) / eps

conv3 = Conv2D(in_channels=3, out_channels=4, kernel_size=3, stride=1, seed=0)
conv3.forward(X)
patches_flat = conv3.patches.reshape(-1, conv3.C_in * conv3.kH * conv3.kW)
dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, conv3.C_out)
analytical_grad = (patches_flat.T @ dout_flat).T.reshape(conv3.W.shape)[i0, i1, i2, i3]

print(f"\nNumerical gradient:  {numerical_grad:.6f}")
print(f"Analytical gradient: {analytical_grad:.6f}")
print(f"Difference: {abs(numerical_grad - analytical_grad):.8f}")