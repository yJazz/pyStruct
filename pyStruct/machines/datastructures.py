from dataclasses import dataclass, field
from itertools import count
import numpy as np
import pandas as pd
from operator import attrgetter
from pyStruct.machines.errors import DataNotExist

    

@dataclass
class BoundaryCondition:
    m_c: float
    m_h: float
    T_c: float 
    T_h: float
    theta_deg: float
    id: int = field(default_factory=count().__next__, init=False)


    def __getitem__(self, item):
        return getattr(self, item)

    def __eq__(self, other):
        names = [key for key in self.__annotations__.keys() if key!='id'] 
        return all([abs(self[name] - other[name]) < 1E-3 for name in names])
    
    def array(self, names: list[str] = None) -> np.array:
        if not names:
            names = self.__annotations__.keys()
        return np.array([self[name] for name in names]).reshape(1, len(names))


@dataclass
class PodFeatures:
    X_spatial:np.ndarray
    X_temporal: np.ndarray
    X_s:np.ndarray
    coord:np.ndarray

@dataclass
class FlowFeatures:
    time_series: np.ndarray # shape: (N_mode, N_t)
    descriptors: np.ndarray # shape: (N_mode, N_features)

# @dataclass
# class Weights:
#     optimize: np.ndarray # shape: (N_mode, 1)
#     predict: np.ndarray # shape: (N_mode, 1)

@dataclass
class Temperature:
    theta_deg: float
    true: np.ndarray # shape: (N_t)
    pred: np.ndarray # shape: (N_t,)
    


class PodSample:
    all_samples = []
    def __init__(self, bc: BoundaryCondition, loc_index: int, wall_true: np.ndarray):
        self.bc = bc
        self.loc_index = loc_index
        self.wall_true = wall_true
        self.name = f'bc{self.bc.id}_theta{bc.theta_deg:.1f}'
        PodSample.all_samples.append(self)

    def set_pod(self, X_spatial:np.ndarray, X_temporal:np.ndarray, X_s:np.ndarray, coord: np.ndarray):
        self.pod = PodFeatures(X_spatial, X_temporal, X_s, coord)

    def set_flowfeatures(self, time_series: np.ndarray, descriptors: np.ndarray):
        self.flow_features = FlowFeatures(time_series, descriptors)

    def set_optimized_weights(self, weights: np.ndarray):
        self.w_optm = weights

    def set_predicted_weights(self, weights: np.ndarray):
        self.w_pred = weights

    def set_wall_pred(self, temperature: np.ndarray):
        self.wall_pred = temperature



# -------------------------------------------------

class PodSets:
    pass

@dataclass
class FlowFeatureSet:
    time_series:np.ndarray #shape: (N_samples, N_mode, N_t)
    descriptors:np.ndarray #shape: (Nsamples, N_mode, N_features)

class WeightsSets:
    pass
class WallSets:
    pass


class PodSampleSet:
    def __init__(self, samples: list[PodSample]):
        self.samples = samples

    def bc(self, names: list[str]=None):
        return np.vstack([sample.bc.array(names) for sample in self.samples])

    @property
    def flow_features(self):
        if any([hasattr(sample, 'flow_features') is False for sample in self.samples] ):
            raise DataNotExist("Need to process `flow_feature`")
        time_series = np.stack([sample.flow_features.time_series for sample in self.samples])
        descriptors = np.stack([sample.flow_features.descriptors for sample in self.samples])
        return FlowFeatureSet(time_series, descriptors)
    
        


