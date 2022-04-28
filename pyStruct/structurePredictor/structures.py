from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


from pyStruct.sampleCollector.sampleStructure import BoundaryCondition


class StructurePredictorInterface(ABC):

    @abstractmethod
    def train(self, structure_table: pd.DataFrame) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def create_model(self):
        raise NotImplementedError()
    
    @abstractmethod
    def save(self, save_to: str) -> None: 
        raise NotImplementedError()

    @abstractmethod
    def load(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, bc: BoundaryCondition) -> np.ndarray:
        raise NotImplementedError()
