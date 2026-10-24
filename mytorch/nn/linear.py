from numpy.random.mtrand import uniform

from ..value import Value
from .module import nnModule

class Linear(nnModule):
    def __init__(self,dim_in=1,dim_out=1) -> None:
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weights = [[Value(uniform(-1,1)) for _ in range(dim_in)]
            for _ in range(dim_out)]
        self.bias = [Value(uniform(-1,1)) for _ in range(dim_out)]

    def forward(self,x):
        outputs = [sum((wi*xi for wi,xi in zip(self.weights[i],x)),self.bias[i]) for i in range(self.dim_out)]
        return outputs

    def parameters(self) -> list[Value]:

        return [
                w for row in self.weights for w in row
            ] + self.bias
