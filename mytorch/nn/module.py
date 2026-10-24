from abc import ABC, abstractmethod

from ..value import Value

class nnModule(ABC):
    @abstractmethod
    def forward(self,x) -> list[Value]:
        pass

    @abstractmethod
    def parameters(self) -> list[Value]:
        return []
