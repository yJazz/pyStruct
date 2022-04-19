import pytest
from dataclasses import dataclass
import hydra
from hydra import compose, initialize
from hydra.errors import MissingConfigException
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from pyStruct.machines.datastructures import BoundaryCondition
from pyStruct.machines.framework import TwoMachineFramework
from pyStruct.machines.config import TwoMachineConfig
from pyStruct.machines.feature_processors import PodCoherentStrength
from pyStruct.machines.structures import GBLookupStructure

@dataclass
class StructureParam:
    x_labels: list[str]
    y_labels: list[str]

@dataclass
class FeatureConfig:
    workspace: str
    N_trains: int
    N_t: int
    N_modes: int
    normalize_y: bool
    theta_degs: list[float]

@pytest.fixture
def feature_processor():
    feature_config = FeatureConfig(
        workspace=r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307',
        N_trains=15,
        N_t=1000,
        N_modes=20,
        normalize_y=False,
        theta_degs=[0, 5, 10, 15]
    )
    return PodCoherentStrength(feature_config)

@pytest.fixture
def bc():
    return BoundaryCondition(m_c=538.8279, m_h=3.72136, T_c=34.95, T_h=88.13)

@pytest.fixture
def non_exist_bc():
    return BoundaryCondition(m_c=538.8279, m_h=3.72136, T_c=34.95, T_h=8)


@pytest.fixture
def machine():
    """ Create a config mimic the behavior of hydra"""
    with initialize(config_path='test_data', job_name='test'):
        cs = ConfigStore.instance()
        cs.store(name='config_good', node=TwoMachineConfig)
        cfg = compose(config_name='config_good')
    machine = TwoMachineFramework(cfg)
    return machine
    
@pytest.fixture
def good_structure_config():
    structure_config = StructureParam(
        x_labels=['m_c', 'vel_ratio'],
        y_labels=['singular', 'spatial']
        )
    return structure_config

@pytest.fixture
def bad_structure_config():
    structure_config = StructureParam(
        x_labels=['m_c', 'v'],
        y_labels=['singular', 'spatial']
        )
    return structure_config

    