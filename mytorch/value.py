import numpy as np

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
