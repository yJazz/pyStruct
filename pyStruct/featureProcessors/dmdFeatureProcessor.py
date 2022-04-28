import numpy as np

from pyStruct.database.datareaders import read_csv
from pyStruct.datastructures.sampleSetStructure import SampleSet
from pyStruct.featureProcessors.featureProcessors import FeatureProcessor, Descriptors, TimeSeries

def standard_dmd(x: np.ndarray, truncate: int = None) -> tuple:
    print("Process dmd")
    x1 = np.array(x)[:, 0:-1]
    x2 = np.array(x)[:, 1:]

    # SVD
    u, s, vH = np.linalg.svd(x1, full_matrices=False)
    v = vH.transpose().conjugate()
    s = np.diag(s)[:truncate, :truncate]
    u = u[:, :truncate]
    v = v[:, :truncate]
    # 
    Atilde = u.transpose() @ x2 @v @np.linalg.inv(s)
    eigs, W = np.linalg.eig(Atilde)

    Phi = x2 @ v @ np.linalg.inv(s) @W

    # Record 
    modes = Phi
    eigenvalues = eigs

    # compute amplitude
    x1_0 = x1[:, 0]
    b = np.linalg.pinv(Phi) @x1_0
    amplitudes = b

    print("Done")
    return modes, eigenvalues, amplitudes

def process_dmd_to_1D_Descriptors() -> np.ndarray:
    pass


class DmdCoherentStrength(FeatureProcessor):
    def __init__(self, feature_config):
        self.feature_config = feature_config

    def _process_sample(self, sample):
        X_matrix = read_csv(sample.X_matrix_path)
        pass
    
    def process_features(self, sample_set: SampleSet) -> tuple[Descriptors, TimeSeries]:
        """ Concrete method on how to obtain descriptors and timeseries"""
        for sample in sample_set.samples:
            pass
        return descriptors, timeseries