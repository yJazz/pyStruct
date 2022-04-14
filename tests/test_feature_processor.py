import pytest
from dataclasses import dataclass
from pyStruct.machines.feature_processors import PodCoherentStrength
from pyStruct.machines.datastructures import *

@dataclass
class FeatureConfig:
    workspace: str
    N_trains: int
    N_t: int
    N_modes: int
    normalize_y: bool
    theta_degs: list[float]

@pytest.fixture
def fp():
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


# ======================================================
def test_1_collect_samples(fp):
    samples = fp.samples
    assert isinstance(samples[0], Sample)

def test_2_get_structure_inputs(fp):
    # correct bc
    structuer_table = fp.get_structure_tables()
    print(structuer_table)