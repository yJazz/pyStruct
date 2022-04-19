""" 
this file defines the configuration structures/hierarchies
"""

from dataclasses import dataclass


@dataclass
class Paths:
    save_to: str

# @dataclass
# class SignacStatepoint:
#     method: str
#     columns: str

@dataclass
class FeaturesParam:
    workspace: str
    N_trains: int
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
    # signac: SignacStatepoint
    machines: Machines
    features: FeaturesParam
    structures: StructureParam

