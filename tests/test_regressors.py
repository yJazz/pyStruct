import pytest
import shutil
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from pyStruct.data.datastructures import PodSampleSet, PodSample
from pyStruct.machines.regressors import GbRegressor

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

class TestGBRegressor:
    output_to = Path('./tests/output/test_regressor/GB')
    @classmethod
    def setup_class(cls):
        cls.output_to.mkdir(parents=True, exist_ok=True)

    @classmethod
    def teardown_class(cls): 
        shutil.rmtree(cls.output_to)
        
    def test_1_train(self, samples_with_w_optms, config):
        training_set = PodSampleSet(samples_with_w_optms)
        machine = GbRegressor(config, TestGBRegressor.output_to)
        machine.train(training_set)
        assert len(machine._models) >0

    def test_2_predict(self, samples_with_w_optms, config):
        training_set = PodSampleSet(samples_with_w_optms)
        machine = GbRegressor(config, TestGBRegressor.output_to)
        machine.train(training_set)
        N_modes_descriptors = training_set.samples[0].flow_features.descriptors
        machine.predict(N_modes_descriptors)



