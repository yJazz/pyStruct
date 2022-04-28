import numpy as np
from pathlib import Path

from pyStruct.database.datareaders import read_csv
from pyStruct.sampleCollector.sampleStructure import Sample
from pyStruct.featureProcessors.featureProcessors import FeatureProcessor, Descriptors, TimeSeries

def standard_pod(x: np.array, truncate: int = None) -> tuple:
    x_mean = x.mean(axis=1)
    x_podmx = x - np.reshape(x_mean, (len(x_mean), 1))
    x_podmx = np.array(x_podmx)
    print("Get Singular & eigenvectors...")
    # SVD: U =  u * np.diag(s) * v
    u, s, v_h = np.linalg.svd(x_podmx, full_matrices=False)
    v = v_h.transpose().conjugate()
    #     s = np.diag(s)[:truncate, :truncate]
    s = s[:truncate]
    u = u[:, :truncate]
    v = v[:, :truncate]

    # Get np from cp
    # u = u.get()  # spatial modes
    # s = s.get()  # singular values
    # v = v.get()  # temporal modes

    spatial_modes = u
    temporal_coeff = v.T
    print("Done")
    return s, spatial_modes, temporal_coeff, x_mean


def process_pod_to_1D_Descriptors(
    loc_index: int, 
    spatials: np.ndarray, 
    singulars: np.ndarray
    ) -> np.ndarray:
    spatial_array = np.linalg.norm(spatials[:, loc_index, :], axis=0)
    return np.vstack((singulars, spatial_array)).T


def process_sample(sample: Sample):
    X_matrix = read_csv(sample.X_matrix_path)[:, -sample.N_t:]
    singulars, spatials, temporals, _ = standard_pod(X_matrix, sample.svd_truncate)
    print(f'Singular shape: {singulars.shape}')
    print(f'Spatial shape: {spatials.shape}')
    print(f'Temporals shape: {temporals.shape}')

    spatials = np.array( np.split(spatials, sample.N_dim, axis=0)) 
    return singulars, spatials, temporals


class PodCoherentStrength(FeatureProcessor):
    def __init__(self, feature_config, folder: Path):
        self.feature_config = feature_config
        self.save_to = folder 

    
    def process_features(self, sample: Sample) -> tuple[Descriptors, TimeSeries]:
        """ Concrete method on how to obtain descriptors and timeseries"""
        saved_feature_files = {
            'descriptors':self.save_to / f'{sample.name}_descriptors.csv',
            'timeseries': self.save_to / f'{sample.name}_timeseries.csv'
        }

        if all([ value.exists() for key, value in saved_feature_files.items()]):
            descriptors = read_csv(saved_feature_files['descriptors'])
            timeseries = read_csv(saved_feature_files['timeseries'])
        else:
            singulars, spatials, temporals = process_sample(sample) 
            descriptors = process_pod_to_1D_Descriptors(
                loc_index = sample.loc_index, 
                spatials=spatials, 
                singulars=singulars
                )
            timeseries = temporals
            np.savetxt(saved_feature_files['descriptors'], descriptors, delimiter=",")
            np.savetxt(saved_feature_files['timeseries'], timeseries, delimiter=",")

        return timeseries, descriptors

