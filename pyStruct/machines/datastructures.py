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
        cond_1 = abs(self.m_c - other.m_c) < 1E-3
        cond_2 = abs(self.m_h - other.m_h) < 1E-3
        cond_3 = abs(self.T_c - other.T_c) < 1E-3
        cond_4 = abs(self.T_h - other.T_h) < 1E-3
        return all([cond_1, cond_2, cond_3, cond_4])




    
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

#---------------------------------
@dataclass
class Wall:
    theta_deg: float
    loc_index: int
    temperature: np.ndarray

@dataclass
class Sample:
    bc: BoundaryCondition
    wall: Wall

@dataclass
class Sample:
    name: str
    bc: BoundaryCondition
    pod: PodFeatures
    walls: list[dict[str, Wall]]
