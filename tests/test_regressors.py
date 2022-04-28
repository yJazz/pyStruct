import pytest
import shutil
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from pyStruct.sampleCollector.sampleSetStructure import SampleSet
from pyStruct.weightsPredictor.gbRegressor import GbRegressor

# Some fixtures
@dataclass
class FeatureConfig:
    N_trains_percent: float
    N_modes: int
    x_labels: list[str]
    y_labels: list[str]

@pytest.fixture
def config():
    return FeatureConfig(
        N_trains_percent=0.8,
        N_modes=20,
        x_labels = ['m_c', 'vel_ratio'],
        y_labels =  ['singular',  'spatial']
    )

@pytest.fixture
def samples_with_w_optms(samples):
    @dataclass
    class FlowFeature:
        descriptors = np.random.rand(20, 5)
        time_series = np.random.rand(20, 1000)

    for sample in samples:
        setattr(sample, 'flow_features', FlowFeature())
        setattr(sample, 'w_optm', np.random.rand(20))
    return samples


class TestGBRegressor:
    output_to = Path('./tests/output/test_regressor/GB')
    @classmethod
    def setup_class(cls):
        cls.output_to.mkdir(parents=True, exist_ok=True)

    @classmethod
    def teardown_class(cls): 
        shutil.rmtree(cls.output_to)
        
    def test_1_train(self, samples_with_w_optms, config):
        training_set = SampleSet(samples_with_w_optms)
        machine = GbRegressor(config, TestGBRegressor.output_to)
        machine.train(training_set)
        assert len(machine._models) >0

    def test_2_predict(self, samples_with_w_optms, config):
        training_set = SampleSet(samples_with_w_optms)
        machine = GbRegressor(config, TestGBRegressor.output_to)
        machine.train(training_set)
        N_modes_descriptors = training_set.samples[0].flow_features.descriptors
        machine.predict(N_modes_descriptors)



