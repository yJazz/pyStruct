import pytest
from pathlib import Path
import shutil
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
from pyStruct.machines.structures import GeneralPredictor
from pyStruct.machines.reconstructors import ReconstructorInterface


@pytest.fixture
def machine():
    """ Create a config mimic the behavior of hydra"""
    with initialize(config_path='test_data', job_name='test'):
        cs = ConfigStore.instance()
        cs.store(name='config_good', node=TwoMachineConfig)
        cfg = compose(config_name='config_good')
    machine = TwoMachineFramework(cfg)
    return machine

def test_1_init_machine(machine):
    print(issubclass(type(machine.feature_processor), FeatureProcessor))
    assert issubclass(type(machine.feature_processor), FeatureProcessor)
    assert issubclass(type(machine.structure_predictor), GeneralPredictor)
    assert issubclass(type(machine.optimizer), Optimizer)
    assert issubclass(machine.weights_predictor, WeightsPredictor)
    assert issubclass(type(machine.reconstructor), ReconstructorInterface)

def test_2_train_structure(machine):
    # clear all data 
    to_folder = Path(machine.config.paths.save_to)/'structure_predictor'
    try:
        shutil.rmtree(to_folder)
    except:
        pass

    # Retrain
    machine._train_structure()
    # # Load
    machine._train_structure()


def test_3_validate_structure(machine):
    machine._train_structure()
    machine._validate_structure()

# def test_4_optimize(machine):
#     machine._optimize()

# def test_5_validate_optimize(machine):
#     samples = machine.validate_optimize()
#     theta_deg = 0
#     y_target = samples[0].walls[f'{theta_deg:.2f}'].temperature
#     y_optm = samples[0].walls_optimized[f'{theta_deg:.2f}']
#     print(f'target shape: {y_target.shape}')
#     print(f'optm shape: {y_optm.shape}')

#     assert y_target.shape== y_optm.shape
