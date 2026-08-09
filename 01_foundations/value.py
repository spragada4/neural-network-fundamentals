class Value:
    """A scalar value that remembers how it was computed, so it can
    compute gradients automatically (a tiny version of what PyTorch's
    autograd does internally)."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data          # the actual number
        self.grad = 0.0           # d(loss)/d(this value) -- filled in later
        self._backward = lambda: None   # how to propagate gradient to children
        self._prev = set(_children)     # which Values created this one
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1  -> chain rule
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(out)/d(self) = other.data, d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort so we process each node AFTER everything
        # that depends on it (reverse order = from output back to inputs)
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = 1.0   # d(self)/d(self) = 1, the starting point
        for v in reversed(topo):
            v._backward()

    # so Python lets you write things like Value(2) - 3, -Value(2), etc.
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"