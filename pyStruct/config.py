""" 
this file defines the configuration structures/hierarchies
"""

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Paths:
    save_to: str


# -----------------------
@dataclass
class FeaturesParam:
    workspace: str
    N_trains_percent: float
    N_t: int
    N_modes: int
    N_dim: int
    normalize_y: bool
    theta_degs: list[float]
    x_labels: list[str]
    y_labels: list[str]

@dataclass
class StructureParam:
    x_labels: list[str]
    y_labels: list[str]


# -------- 

@dataclass
class Machines:
    feature_processor: str
    optimizer: str
    structure_predictor: str
    weights_predictor: str
    reconstructor: str

@dataclass
class TwoMachineConfig:
    paths: Paths
    machines: Machines
    features: FeaturesParam


def check_config(config):
    assert abs(int(config.features.N_trains_percent)) <=1

    if config.machines.optimizer == 'INTERCEPT':
        assert config.machines.reconstructor == 'INTERCEPT'
    if config.machines.reconstructor =='INTERCEPT':
        assert config.machines.optimizer =='INTERCEPT'
    return
