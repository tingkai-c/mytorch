import numpy as np
from random import randint, random

class Value:
    def __init__(self, data,_op=None,_prev=()):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._op = _op
        self._prev = set(_prev)

    def __repr__(self) -> str:
        return "val: "+ str(self.data) + ", grad:"+ str(self.grad)

    def __add__(self,other: "Value | int | float"):
        if not isinstance(other, (Value, int, float)):
            return NotImplemented
        other = other if isinstance(other, Value) else Value(other)
        output =  Value(data=self.data+other.data,_op="+",_prev=(self,other))

        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward
        return output

    def __radd__(self, other: "Value | int | float"):
        return self + other

    def __mul__(self,other: "Value | int | float"):
        if not isinstance(other, (Value, int, float)):
            return NotImplemented
        other = other if isinstance(other,Value) else Value(other)
        output =  Value(data=self.data * other.data,_op="*",_prev =(self,other))

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward
        return output

    def __rmul__(self, other: "Value | int | float"):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other: "Value | int | float"):
        if not isinstance(other, (Value, int, float)):
            return NotImplemented
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other: "Value | int | float"):
        if not isinstance(other, (Value, int, float)):
            return NotImplemented
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __pow__(self,exponent: int | float):
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        output = Value(data=self.data**exponent,_op="pow",_prev=(self,))

        def _backward():
            self.grad += exponent * self.data**(exponent-1) * output.grad
        output._backward = _backward

        return output

    def sigmoid(self):
        value = 1 / (1 + pow(np.e, -self.data))

        output = Value(data=value, _op="sigmoid", _prev=(self,))

        def _backward():
            self.grad += value * (1-value) * output.grad

        output._backward = _backward
        return output


    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zerograd(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
        build_topo(self)

        self.grad = 0
        for node in reversed(topo):
            node.grad = 0

class Neuron:
    def __init__(self,dim=1) -> None:
        self.weights = [Value(randint(-1,1)) for _ in range(dim)]
        self.bias = Value(randint(-1,1))

    def parameters(self):
        return self.weights + [self.bias]

    def forward(self,x):
        activation = sum((wi*xi for wi,xi in zip(self.weights,x)),self.bias)
        out = activation.sigmoid()
        return out


class Layer:
    def __init__(self,dim_in=1,dim_out=1) -> None:
        self.neurons = [Neuron(dim=dim_in) for _ in range(dim_out)]

    def forward(self,x):
        outs = [neuron.forward(x) for neuron in self.neurons]
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]



class MLP:
    def __init__(self,dims: list[int]) -> None:
        self.layers = [Layer(dims[i],dims[i+1]) for i in range(len(dims) - 1)]

    def forward(self,x):
        cur = x
        for layer in self.layers:
            cur = layer.forward(cur)
        return cur

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



x = [1,2,3,4]
target = Value(0.427)

learning_rate = 0.05
epochs = 200
tests = 10
for test in range(tests):
    mlp = MLP([4,48,48,1])
    for epoch in range(epochs):

        loss = (mlp.forward(x)[0] - target) ** 2
        loss.zerograd()
        loss.backward()

        for param in mlp.parameters():
            param.data -= param.grad * learning_rate


    print(mlp.forward(x))
