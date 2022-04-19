from dataclasses import dataclass
from pyStruct.machines.feature_processors import PodCoherentStrength
from pyStruct.machines.datastructures import *




# ======================================================
def test_1_collect_samples(feature_processor):
    samples = feature_processor.samples
    assert isinstance(samples[0], Sample)

def test_2_get_structure_inputs(feature_processor):
    # correct bc
    structure_table = feature_processor.get_structure_tables()
    structure_table.to_csv('structure_table.csv', index=False)
    # print(structure_table)

def test_3_compose_temporal_matrix(feature_processor):
    samples = feature_processor.training_samples
    assert isinstance(samples[0] , Sample)
    X_samples = feature_processor.compose_temporal_matrix(samples)
    N_modes = feature_processor.feature_config.N_modes
    N_t = feature_processor.feature_config.N_t
    assert X_samples.shape == (len(samples), N_modes, N_t)

def test_4_get_temporal_signals_outputs(feature_processor):
    theta_deg = 0
    samples = feature_processor.training_samples
    print(samples[0].walls[f'{theta_deg:.2f}'].temperature.shape)
    y_samples = feature_processor.get_temporal_signals_outputs(samples, theta_deg=0)
    N_t = feature_processor.feature_config.N_t
    assert y_samples.shape ==(len(samples), N_t)