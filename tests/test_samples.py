import numpy as np

from dataclasses import dataclass
from pyStruct.sampleCollector.samples import initialize_sample_from_signac
from pyStruct.sampleCollector.sampleStructure import Sample, BoundaryCondition
from pyStruct.sampleCollector.sampleSetStructure import SampleSet


@dataclass
class SignacSample:
  processor_workspace = r'F:\project2_phD_bwrx\db_bwrx_processor'
  parent_workspace= r'F:\project2_phD_bwrx\db_bwrx_cfd'
  columns= ['Velocity[i] (m/s)','Velocity[j] (m/s)', 'Velocity[k] (m/s)']
  method= 'dmd'
  N_t= 1000
  dt= 0.005
  svd_truncate= 20
  normalize_y= False
  bc_mapper={
    'M_COLD_KGM3S': 'm_c',
    'M_HOT_KGM3S': 'm_h',
    'T_COLD_C': 'T_c',
    'T_HOT_C': 'T_h'
  }
  theta_degs= [0]


def test_generate_samples():
    config = SignacSample()
    samples = initialize_sample_from_signac(config)
    assert len(samples) > 0 
    assert all([isinstance(s, Sample) for s in samples])

class TestSample:
    def setup(self):
        config = SignacSample()
        self.sample = initialize_sample_from_signac(config)[0]
    def test_bc(self):
        print(self.sample.bc)
        assert isinstance(self.sample.bc, BoundaryCondition)

    
class TestSampleSet:
    def setup(self):
        config = SignacSample()
        samples = initialize_sample_from_signac(config)
        self.sample_set = SampleSet(samples)
    def test_get_bc(self):
        bcs = self.sample_set.bc()
        assert type(bcs) == np.ndarray

