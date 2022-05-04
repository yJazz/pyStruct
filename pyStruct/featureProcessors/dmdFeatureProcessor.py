import numpy as np
from pathlib import Path

from pyStruct.database.datareaders import read_csv, save_csv, read_csv_complex
from pyStruct.sampleCollector.sampleStructure import Sample
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



def process_sample(sample: Sample, processor_files: dict):
    X_matrix = read_csv(sample.X_matrix_path)[:, -sample.N_t:]
    modes, eigenvalues, amplitudes= standard_dmd(X_matrix, sample.svd_truncate)
    # save file
    save_processor_files(modes, eigenvalues, amplitudes, processor_files)
    return modes, eigenvalues, amplitudes

def save_processor_files(modes, eigenvalues, amplitudes, processor_files):
    save_csv(processor_files['modes'], modes)
    save_csv(processor_files['eigenvalues'], eigenvalues)
    save_csv(processor_files['amplitudes'], amplitudes)

def read_processor_files(processor_files: dict):
    modes = read_csv_complex(processor_files['modes'])
    eigenvalues = read_csv_complex(processor_files['eigenvalues'])
    amplitudes = read_csv_complex(processor_files['amplitudes'])
    return modes, eigenvalues, amplitudes


def convert_eigenvalue_to_frequency(eigenvalue: complex, dt: float):
    return np.log(eigenvalue).imag /(2*np.pi*dt)

def process_dmd_to_1D_Descriptors(
    loc_index: int,
    modes: np.ndarray,
    amplitudes: np.ndarray
    ) -> np.ndarray:
    pass

def process_dmd_to_timeseries():
    pass


class DmdCoherentStrength(FeatureProcessor):
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
            # Check if processor files exists 
            processor_files = {
                'modes': self.save_to / f'{sample.name}_modes.csv',
                'eigenvalues': self.save_to / f'{sample.name}_eigenvalues.csv',
                'amplitudes': self.save_to / f'{sample.name}_amplitudes.csv',
            }
            # check files 
            if all([value.exists() for key, value in processor_files.items()]):
                modes, eigenvalues, amplitudes = read_processor_files(processor_files)

            else:
                modes, eigenvalues, amplitudes = process_sample(sample, processor_files) 

            modes = np.array( np.split(modes, sample.N_dim, axis=0)) 
            # descriptors = process_dmd_to_1D_Descriptors(
            #     loc_index = sample.loc_index, 
            #     modes = modes,
            #     amplitudes = amplitudes,
            #     )
            # timeseries = process_dmd_to_timeseries(eigenvalues)
            # save_csv(saved_feature_files['descriptors'], descriptors)
            # save_csv(saved_feature_files['timeseries'], timeseries)

        return timeseries, descriptors