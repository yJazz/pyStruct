import pytest
from dataclasses import dataclass, fields
from pathlib import Path
from pyStruct.directory import *

@dataclass
class MachineConfig:
    feature_processor: str
    optimizer: str
    structure_predictor: str
    weights_predictor: str
    reconstructor: str

    def items(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}.items()
        

@pytest.fixture
def machine_config():
    return MachineConfig(
        'COHERENT_STRENGTH',
        'ALL_WEIGHTS',
        'GB',
        'GB',
        'LINEA'
    )
        
def test_1_test_init(machine_config):
    root =  r'F:\project2_phD_bwrx\prediction_data\20220420_refactor'
    paths = FrameworkPaths(root, machine_config)
    assert hasattr(paths, 'feature_processor')
