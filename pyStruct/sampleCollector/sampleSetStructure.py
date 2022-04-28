from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from pyStruct.sampleCollector.sampleStructure import Sample
from pyStruct.errors import DataNotExist


class SampleSet(ABC):
    @abstractmethod
    def bc(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def flow_features(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def pred_structures_names(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def pred_descriptors(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def w_optm(self):
        raise NotImplementedError


@dataclass
class FlowFeatureSet:
    time_series:np.ndarray #shape: (N_samples, N_mode, N_t)
    descriptors:np.ndarray #shape: (Nsamples, N_mode, N_features)


class SampleSet(SampleSet):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def bc(self, names: list[str]=None):
        return np.vstack([sample.bc.array(names) for sample in self.samples])
    
    @property
    def name(self):
        return [sample.name for sample in self.samples]

    @property
    def flow_features(self):
        """
        time_series shape: (N_samples, N_modes, N_t)
        descriptors shape: (N_samples, N_modes, N_features)
        """
        if any([hasattr(sample, 'flow_features') is False for sample in self.samples] ):
            raise DataNotExist("Need to process `flow_feature`")
        time_series = np.stack([sample.flow_features.time_series for sample in self.samples])
        descriptors = np.stack([sample.flow_features.descriptors for sample in self.samples])
        return FlowFeatureSet(time_series, descriptors)
    
    @property
    def pred_structures_names(self):
        if any([hasattr(sample, 'pred_structures_names') is False for sample in self.samples] ):
            raise DataNotExist("Need to run `train_structures`")
        return [sample.pred_structures_names for sample in self.samples]
    
    @property
    def pred_descriptors(self):
        """
        shape (N_samples, N_modes, N_features)
        """
        if any([hasattr(sample, 'pred_descriptors') is False for sample in self.samples] ):
            raise DataNotExist("")
        return np.stack([sample.pred_descriptors for sample in self.samples])

    
    @property
    def w_optm(self):
        """ shape: (N_samples, N_modes, 1)"""
        if any([hasattr(sample, 'w_optm') is False for sample in self.samples] ):
            raise DataNotExist("Need to do optimization")
        return np.stack([sample.w_optm for sample in self.samples])

