from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class BoundaryCondition:
    m_c: float
    m_h: float
    T_c: float 
    T_h: float
    vel_ratio: float = field(init=False)

    def __post_init__(self):
        self.vel_ratio = self.m_c / self.m_h

    def __getitem__(self, item):
        return getattr(self, item)

    def __eq__(self, other):
        names = self.__annotations__.keys()
        return all([abs(self[name] - other[name]) < 1E-3 for name in names])
    
    def array(self, names: list[str] = None) -> np.array:
        if not names:
            names = self.__annotations__.keys()
        return np.array([self[name] for name in names]).reshape(1, len(names))

# ------------------------------
@dataclass
class Features:
    pass

@dataclass
class PodFeatures(Features):
    X_spatial:np.ndarray
    X_temporal: np.ndarray
    X_s:np.ndarray
    coord:np.ndarray

@dataclass
class DmdFeatures(Features):
    pass

@dataclass
class ProcessedFeatures:
    pass

@dataclass
class PodFeaturesDescriptors(ProcessedFeatures):
    singular: np.ndarray
    spatial: np.array


#---------------------------------
@dataclass
class Wall:
    theta_deg: float
    loc_index: int
    temperature: np.ndarray

@dataclass
class Sample:
    name: str
    bc: BoundaryCondition
    pod: PodFeatures
    walls: list[dict[str, Wall]]
    # To be computed 
    weights_optimized: list[dict[str, np.ndarray]] = field(default_factory=lambda: {})
    walls_optimized: list[dict[str, np.ndarray]] = field(default_factory=lambda: {})
    # weights_pred: list[dict[str, np.ndarray]] = field(default_factory=dict)
    # walls_pred: list[dict[str, np.ndarray]] = field(default_factory=dict)
    
    def record_optimize(self, theta_deg: float, weights:np.ndarray, y_optm: np.ndarray):
        self.weights_optimized[f'{theta_deg:.2f}'] = weights
        self.walls_optimized[f'{theta_deg:.2f}'] = y_optm 
        return

    def record_prediction(self, theta_deg: float, weights:np.ndarray, y_pred: np.ndarray):
        self.weights_pred[f'{theta_deg:.2f}'] = weights
        self.walls_pred[f'{theta_deg:.2f}'] = y_pred 
        return




