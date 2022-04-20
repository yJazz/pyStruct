import pytest
from dataclasses import dataclass
from pyStruct.machines.feature_processors import PodCoherentStrength
from pyStruct.data.datastructures import *
from pyStruct.data.dataset import PodModesManager


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

@pytest.fixture
def feature_processor(config):
    return PodCoherentStrength(config)

@pytest.fixture
def samples(config):
    """ A messy code... should refactor later"""
    workspace = config.workspace
    theta_degs = config.theta_degs
    normalize_y = config.normalize_y
    samples = []

    pod_manager = PodModesManager(name='', workspace=workspace, normalize_y=normalize_y)
    bcs = pod_manager._read_signac()
    for sample in range(pod_manager.N_samples):
        m_c = bcs[sample]['M_COLD_KGM3S']
        m_h = bcs[sample]['M_HOT_KGM3S']
        T_c = bcs[sample]['T_COLD_C']
        T_h = bcs[sample]['T_HOT_C']

        X_spatial = pod_manager.X_spatials[sample, ...]
        X_temporal = pod_manager.X_temporals[sample, ...]
        X_s = pod_manager.X_s[sample, ...]
        coord = pod_manager.coords[sample]
        # pod = PodFeatures(coord=coord, X_spatial=X_spatial, X_temporal=X_temporal, X_s=X_s)

        for theta_deg in theta_degs:
            T_wall = pod_manager.T_walls[f'{theta_deg:.2f}']['T_wall'][sample, :]
            loc_index =pod_manager.T_walls[f'{theta_deg:.2f}']['loc_index']
            bc = BoundaryCondition(m_c, m_h, T_c, T_h, theta_deg)
            s = PodSample(bc, loc_index, T_wall)
            s.set_pod(X_spatial, X_temporal, X_s, coord)
            samples.append(s)
    return samples

# ======================================================
def test_1_process_samples_and_modify_the_original_sample(feature_processor, samples):
    # Process single sample
    sample_set = PodSampleSet(samples[0:1])
    assert hasattr(samples[0], 'flow_features') is False
    feature_processor.process_samples(sample_set)
    assert len(samples) >1
    assert hasattr(samples[0], 'flow_features')
    assert hasattr(samples[1], 'flow_features') is False


    

# def test_2_get_structure_training_pairs(feature_processor):
#     X, y = feature_processor.get_structure_training_pairs()
#     x_labels = feature_processor.feature_config.x_labels
#     y_labels = feature_processor.feature_config.y_labels
#     N_modes = feature_processor.feature_config.N_modes
#     samples = feature_processor.training_samples
#     assert X.shape == (len(samples), len(x_labels))
#     assert y.shape == (len(samples), N_modes, len(y_labels))

# def test_2_get_structure_inputs(feature_processor):
#     # correct bc
#     structure_table = feature_processor.get_structure_tables()
#     structure_table.to_csv('structure_table.csv', index=False)
#     # print(structure_table)

# def test_3_compose_temporal_matrix(feature_processor):
#     samples = feature_processor.training_samples
#     assert isinstance(samples[0] , Sample)
#     X_samples = feature_processor.compose_temporal_matrix(samples)
#     N_modes = feature_processor.feature_config.N_modes
#     N_t = feature_processor.feature_config.N_t
#     assert X_samples.shape == (len(samples), N_modes, N_t)

# def test_4_get_temporal_signals_outputs(feature_processor):
#     theta_deg = 0
#     samples = feature_processor.training_samples
#     print(samples[0].walls[f'{theta_deg:.2f}'].temperature.shape)
#     y_samples = feature_processor.get_temporal_signals_outputs(samples, theta_deg=0)
#     N_t = feature_processor.feature_config.N_t
#     assert y_samples.shape ==(len(samples), N_t)