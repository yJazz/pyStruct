""" 

"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

from pyStruct.database.datareaders import find_loc_index

# ----------------- BoundaryCondition ---------------
@dataclass
class BoundaryCondition:
    m_c: float
    m_h: float
    T_c: float 
    T_h: float
    theta_deg: float
    id: int = field(init=False)

    def __post_init__(self):
        self.id = f'{self.m_c/self.m_h:.1f}' 

    def __getitem__(self, item):
        return getattr(self, item)

    def __eq__(self, other):
        names = [key for key in self.__annotations__.keys() if key!='id'] 
        return all([abs(self[name] - other[name]) < 1E-3 for name in names])
    
    def array(self, names: list[str] = None) -> np.array:
        if not names:
            names = self.__annotations__.keys()
        return np.array([self[name] for name in names]).reshape(1, len(names))


# ------- Sample --------------------------------

@dataclass
class PodFeatures:
    X_spatial:np.ndarray
    X_temporal: np.ndarray
    X_s:np.ndarray

@dataclass
class FlowFeatures:
    time_series: np.ndarray # shape: (N_mode, N_t)
    descriptors: np.ndarray # shape: (N_mode, N_features)

@dataclass
class Temperature:
    theta_deg: float
    true: np.ndarray # shape: (N_t)
    pred: np.ndarray # shape: (N_t,)


class SampleInterface(ABC):
    @abstractmethod
    def set_flowfeatures(self):
        raise NotImplementedError()

    @abstractmethod
    def set_pred_structures(self):
        raise NotImplementedError()
    
    @abstractmethod
    def set_pred_descriptors(self):
        raise NotImplementedError()

    @abstractmethod
    def set_optimized_weights(self):
        raise NotImplementedError()

    @abstractmethod
    def set_predicted_weights(self):
        raise NotImplementedError()

    @abstractmethod
    def set_wall_pred(self):
        raise NotImplementedError()



class Sample(SampleInterface):
    """ A concrete class"""
    def __init__(self,
        bc: BoundaryCondition, 
        N_dim: int,
        N_t: int,
        svd_truncate: int,
        dt: float,
        coord: pd.DataFrame, 
        wall_true: np.ndarray,
        X_matrix_path: Path,
        ):

        self.bc = bc
        self.N_dim = N_dim
        self.N_t = N_t
        self.svd_truncate = svd_truncate
        self.dt = dt
        self.coord = coord
        self.wall_true = wall_true
        self.X_matrix_path = X_matrix_path

        self.name = f'bc{self.bc.id}_theta{bc.theta_deg:.1f}'
        self.loc_index = find_loc_index(bc.theta_deg, coord)

    def set_flowfeatures(self, time_series: np.ndarray, descriptors: np.ndarray):
        self.flow_features = FlowFeatures(time_series, descriptors)
    
    def set_pred_structures(self, pred_structures_names: list[str]):
        self.pred_structures_names = pred_structures_names
    
    def set_pred_descriptors(self, descriptors: np.ndarray):
        self.pred_descriptors = descriptors

    def set_optimized_weights(self, weights: np.ndarray):
        self.w_optm = weights

    def set_predicted_weights(self, weights: np.ndarray):
        self.w_pred = weights

    def set_wall_pred(self, temperature: np.ndarray):
        self.wall_pred = temperature
