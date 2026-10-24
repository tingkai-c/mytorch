from .module import nnModule

class Sequential(nnModule):
    def __init__(self,*modules: nnModule) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x):
        cur = x
        for layer in self.modules:
            cur = layer.forward(cur)

        return cur

    def parameters(self):
        return [param for module in self.modules for param in module.parameters()]
