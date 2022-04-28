from abc import ABC, abstractmethod

from typing import Protocol
from numpy.typing import ArrayLike

from pyStruct.sampleCollector.sampleStructure import Sample
 
Descriptors = ArrayLike
TimeSeries = ArrayLike

class FeatureProcessor(ABC):
    @abstractmethod
    def process_features(self, sample: Sample) -> tuple[Descriptors, TimeSeries]:
        raise NotImplementedError()
