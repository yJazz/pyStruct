import pytest
import hydra

from hydra import compose, initialize
from hydra.errors import MissingConfigException
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from pyStruct.machines.config import TwoMachineConfig
from pyStruct.machines.framework import TwoMachineFramework
from pyStruct.machines.feature_processors import FeatureProcessor
from pyStruct.machines.optimizers import Optimizer
from pyStruct.machines.regressors import WeightsPredictor
from pyStruct.machines.structures import StructurePredictor

@dataclass
class Machines:
    feature_processor: str
    optimizer: str
    structure_predictor: str
    weights_predictor: str

@dataclass
class Config:
    machines: Machines

@pytest.fixture
def machine():
    """ Create a config mimic the behavior of hydra"""
    with initialize(config_path='config', job_name='test'):
        cs = ConfigStore.instance()
        cs.store(name='config_good', node=TwoMachineConfig)
        cfg = compose(config_name='config_good')
    machine = TwoMachineFramework(cfg)
    return machine


def test_1_init_machine(machine):
    assert issubclass(machine.feature_processor, FeatureProcessor)
    assert issubclass(machine.optimizer, Optimizer)
    assert issubclass(machine.structure_predictor, StructurePredictor)
    assert issubclass(machine.weights_predictor, WeightsPredictor)

def test_2_train_structure(machine):
    machine._train_structure()

