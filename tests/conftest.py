import pytest
from dataclasses import dataclass
from pyStruct.sampleCollector.samples import initialize_sample_from_signac
import pandas as pd
import numpy as np

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

@pytest.fixture
def samples():
    config = SignacSample()
    return initialize_sample_from_signac(config)
