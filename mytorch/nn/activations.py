from ..value import Value
from .module import nnModule

class Sigmoid(nnModule):

    def forward(self, x) -> list[Value]:
        outputs = [xi.sigmoid() for xi in x]
        return outputs

    def parameters(self) -> list[Value]:
        return []
