from value import Value

# f(x, y) = x*y + x**2
x = Value(3.0)
y = Value(-2.0)
f = x * y + x**2

f.backward()

print("f.data =", f.data)     # expect 3*-2 + 9 = 3.0
print("df/dx  =", x.grad)     # expect y + 2x = -2 + 6 = 4.0
print("df/dy  =", y.grad)     # expect x = 3.0