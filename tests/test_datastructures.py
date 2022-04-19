import pytest
from dataclasses import dataclass
from pyStruct.data.dataset import PodModesManager
from pyStruct.machines.datastructures import *

 
def test_1_test_eq():
    bc_1 = BoundaryCondition(m_c=1, m_h=1, T_c=1, T_h=1, theta_deg=0)
    bc_2 = BoundaryCondition(m_c=1, m_h=1, T_c=1, T_h=1, theta_deg=0)
    assert (bc_1 == bc_2) 

    # small floating error, still same
    bc_3 = BoundaryCondition(m_c=1.00001, m_h=1, T_c=1, T_h=1, theta_deg=0)
    assert (bc_1 == bc_3)

    # different bc
    bc_4 = BoundaryCondition(m_c=5, m_h=3, T_c=1, T_h=1, theta_deg=0)
    assert (bc_1 == bc_4) is False

def test_1_PodSampleSet_get_bc(samples):
    sample_set = PodSampleSet(samples)
    bc = sample_set.bc
    assert bc.shape == (len(samples), samples[0].bc.array().shape[-1])

def test_2_PodSampleSet_get_flow_features(samples):
    samples = samples[0:3]
    # Process data
    N_t = 10
    N_features = 4
    N_modes = 20
    time_series = np.random.rand(N_modes, N_t)
    descriptor = np.random.rand(N_modes, N_features)
    for sample in samples:
        sample.set_flowfeatures(time_series, descriptor)

    sample_set = PodSampleSet(samples)
    aggre_time_series = sample_set.flow_features.time_series
    aggre_descriptors = sample_set.flow_features.descriptors

    assert aggre_time_series.shape == (len(samples), N_modes, N_t)
    assert aggre_descriptors.shape == (len(samples), N_modes, N_features)
    
def test_3_PodSampleSet_get_flow_features_not_exist(samples):
    sample_set = PodSampleSet(samples)
    try:
        aggre_time_series = sample_set.flow_features.time_series
    except DataNotExist:
        pass 
    except:
        assert False
