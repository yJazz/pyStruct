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
    N_trains_percent: float
    N_modes: int
    theta_degs: list[float]
    x_labels: list[str]
    y_labels: list[str]

@dataclass
class StructureParam:
    x_labels: list[str]
    y_labels: list[str]

@dataclass
class BcMapper:
    M_COLD_KGM3S: str 
    M_HOT_KGM3S: str
    T_COLD_C: str
    T_HOT_C: str


@dataclass 
class SignacSample:
    processor_workspace: str
    parent_workspace: str
    columns: list[str]
    method: str
    N_t: int
    dt: float
    svd_truncate: int
    normalize_y: bool
    bc_mapper: BcMapper
    theta_degs: list[float]

# -------- 

@dataclass
class Machines:
    feature_processor: str
    optimizer: str
    structure_predictor: str
    weights_predictor: str
    reconstructor: str

# -------- 
@dataclass
class TwoMachineConfig:
    paths: Paths
    signac_sample: SignacSample
    machines: Machines
    features: FeaturesParam


def check_config(config):
    assert abs(int(config.features.N_trains_percent)) <=1

    if config.machines.optimizer == 'INTERCEPT':
        assert config.machines.reconstructor == 'INTERCEPT'
    if config.machines.reconstructor =='INTERCEPT':
        assert config.machines.optimizer =='INTERCEPT'
    return
