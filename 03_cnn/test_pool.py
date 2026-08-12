import numpy as np
from pool_layer import MaxPool2D

X = np.array([[[
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [0, 1, 9, 3],
    [2, 4, 5, 7],
]]], dtype=float)   # shape (1, 1, 4, 4)

pool = MaxPool2D(size=2, stride=2)
out = pool.forward(X)
print("Output:\n", out[0, 0])
# Expect: top-left window max(1,3,5,6)=6, top-right max(2,4,1,2)=4
#         bottom-left max(0,1,2,4)=4, bottom-right max(9,3,5,7)=9

dout = np.ones_like(out)
dX = pool.backward(dout)
print("\ndX (gradient only at max positions):\n", dX[0, 0])