import random
from value import Value


class Neuron:
    def __init__(self, n_inputs, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act   # <-- only apply ReLU if nonlin

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_inputs, n_neurons, nonlin=True):
        self.neurons = [Neuron(n_inputs, nonlin=nonlin) for _ in range(n_neurons)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_inputs, layer_sizes):
        sizes = [n_inputs] + layer_sizes
        self.layers = []
        for i in range(len(layer_sizes)):
            is_last = (i == len(layer_sizes) - 1)
            self.layers.append(Layer(sizes[i], sizes[i+1], nonlin=not is_last))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]