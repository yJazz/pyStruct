from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def optimize(self):
        raise NotImplementedError
