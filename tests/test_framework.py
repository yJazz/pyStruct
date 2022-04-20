import pytest
from pathlib import Path
import shutil
import hydra

from hydra import compose, initialize
from hydra.errors import MissingConfigException
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from pyStruct.config import TwoMachineConfig
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
    assert issubclass(type(machine.weights_predictor), WeightsPredictor)
    assert issubclass(type(machine.reconstructor), ReconstructorInterface)

def test_2_train_structure_new_train(machine):
    # clear all data 
    try:
        shutil.rmtree(machine.paths.structure_predictor / 'model.pkl')
    except:
        pass
    # Retrain
    machine._train_structure()
def test_3_train_structure_load(machine):
    # # Load
    machine._train_structure()

def test_4_optimize(machine):
    config = machine.config
    N_samples = config.features.N_modes
    assert all( [hasattr(sample, 'w_optm') is False for sample in machine.training_set.samples])
    machine._optimize()
    assert all( [hasattr(sample, 'w_optm') for sample in machine.training_set.samples])
    assert machine.training_set.w_optm.shape == (len(machine.training_set.samples), N_samples)

def test_5_train_and_predict(machine):
    machine.train()
    sample = machine.training_set.samples[0]
    machine.predict(sample.bc)
