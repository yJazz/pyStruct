import pytest
from dataclasses import dataclass
from pyStruct.featureProcessors.podFeatureProcessor import PodCoherentStrength
from pyStruct.sampleCollector.sampleSetStructure import SampleSet


@pytest.fixture
def feature_processor(config):
    return PodCoherentStrength(config)

@dataclass
class Feature:
  N_trains_percent= 0.8
  N_modes= 20
  x_labels= ['m_c', 'm_h', 'T_c', 'T_h', 'theta_deg']
  y_labels= ['singular',  'spatial']

class TestPodCoherentStrength:
    def setup(self):
        self.config = Feature()
        self.feature_processor = PodCoherentStrength(self.config)

    def test_process_features(self, samples):
        self.sample = samples[0] 
        descriptors, timeseries = self.feature_processor.process_features(self.sample)
        assert descriptors.shape == (self.config.N_modes, len(self.config.y_labels))