import pytest
from dataclasses import dataclass
import hydra
from hydra import compose, initialize
from hydra.errors import MissingConfigException
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from pyStruct.machines.datastructures import BoundaryCondition, PodSample
from pyStruct.machines.framework import TwoMachineFramework
from pyStruct.machines.config import TwoMachineConfig
from pyStruct.machines.feature_processors_depr import PodCoherentStrength
from pyStruct.machines.structures import GBLookupStructure
from pyStruct.data.dataset import PodModesManager


# Some fixtures
@dataclass
class FeatureConfig:
    workspace: str
    N_trains: int
    N_t: int
    N_modes: int
    normalize_y: bool
    theta_degs: list[float]
    x_labels: list[str]
    y_labels: list[str]

@pytest.fixture
def config():
    return FeatureConfig(
        workspace=r'F:\project2_phD_bwrx\db_bwrx_aggregation\workspace\4ff1b3520e44de10cea74136b19eb307',
        N_trains=15,
        N_t=1000,
        N_modes=20,
        normalize_y=False,
        theta_degs=[0, 5],
        x_labels = ['m_c', 'vel_ratio'],
        y_labels =  ['singular',  'spatial']
    )

@pytest.fixture
def samples(config):
    """ A messy code... should refactor later"""
    workspace = config.workspace
    theta_degs = config.theta_degs
    normalize_y = config.normalize_y

    samples=[]
    pod_manager = PodModesManager(name='', workspace=workspace, normalize_y=normalize_y)
    bcs = pod_manager._read_signac()
    for sample in range(pod_manager.N_samples):
        m_c = bcs[sample]['M_COLD_KGM3S']
        m_h = bcs[sample]['M_HOT_KGM3S']
        T_c = bcs[sample]['T_COLD_C']
        T_h = bcs[sample]['T_HOT_C']

        X_spatial = pod_manager.X_spatials[sample, ...]
        X_temporal = pod_manager.X_temporals[sample, ...]
        X_s = pod_manager.X_s[sample, ...]
        coord = pod_manager.coords[sample]
        # pod = PodFeatures(coord=coord, X_spatial=X_spatial, X_temporal=X_temporal, X_s=X_s)

        for theta_deg in theta_degs:
            T_wall = pod_manager.T_walls[f'{theta_deg:.2f}']['T_wall'][sample, :]
            loc_index =pod_manager.T_walls[f'{theta_deg:.2f}']['loc_index']
            bc = BoundaryCondition(m_c, m_h, T_c, T_h, theta_deg)
            s = PodSample(bc, loc_index, T_wall)
            s.set_pod(X_spatial, X_temporal, X_s, coord)
            samples.append(s)
    return samples



@dataclass
class StructureParam:
    x_labels: list[str]
    y_labels: list[str]


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
        x_labels=['m_c', 'm_h', 'T_c', 'T_h', 'theta_deg'],
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

    