from typing import Callable
import numpy as np
from pathlib import Path

from skimage.metrics import structural_similarity as ssim

from pyStruct.database.datareaders import read_csv, save_csv
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
    s = s[:truncate].reshape(-1, 1)
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
    """
    output shape: (N_modes, 2)
    """
    spatial_array = np.linalg.norm(spatials[:, loc_index, :], axis=0).reshape(-1, 1)
    return np.hstack((singulars, spatial_array))

def process_pod_to_1D_Descriptors_SSIM(
    spatials: np.ndarray, 
    spatials_ref: np.ndarray, 
    singulars: np.ndarray
    )-> np.ndarray:
    """
    output shape: (N_modes, 2)
    """
    ssim_output = np.zeros((spatials.shape[0], spatials.shape[2]))
    for d in range(spatials.shape[0]):
        for mode in range(spatials.shape[-1]):
            index = ssim(
                spatials[d, :, mode], 
                spatials_ref[d, :, mode], 
                data_range = spatials[d, :, mode].max() - spatials[d, :, mode].min(), 
                )
            ssim_output[d, mode] = index
    spatial_array = np.linalg.norm(ssim_output, axis=0).reshape(-1, 1)
    return np.hstack((singulars, spatial_array))


def process_sample(sample: Sample):
    X_matrix = read_csv(sample.X_matrix_path)[:, -sample.N_t:]
    singulars, spatials, temporals, _ = standard_pod(X_matrix, sample.svd_truncate)
    print(f'Singular shape: {singulars.shape}')
    print(f'Spatial shape: {spatials.shape}')
    print(f'Temporals shape: {temporals.shape}')
    return singulars, spatials, temporals


def read_or_process_pod_sample(sample: Sample, feature_processor_directory: str):
    files = {
        'spatials': Path(feature_processor_directory) /'pod' / f"spatial_{str(sample.bc.id).replace('.', 'p')}.csv" ,
        'temporals': Path(feature_processor_directory) /'pod' / f"temporals_{str(sample.bc.id).replace('.', 'p')}.csv" ,
        'singulars': Path(feature_processor_directory) /'pod' / f"singulars_{str(sample.bc.id).replace('.', 'p')}.csv" ,
    }
    files_exist = all([ value.exists() for key, value in files.items()])
    print(files['spatials'])
    print(files['temporals'])
    print(files['singulars'])
    if files_exist:
        print("POD files found; LOAD")
        singulars = read_csv(files['singulars'])
        spatials = read_csv(files['spatials'])
        temporals = read_csv(files['temporals'])
    else:
        singulars, spatials, temporals = process_sample(sample)
        # save
        folder = Path(feature_processor_directory)/'pod'
        folder.mkdir(parents=True, exist_ok=True)
        save_csv(files['singulars'], singulars)
        save_csv(files['spatials'], spatials)
        save_csv(files['temporals'], temporals)
    spatials = np.array( np.split(spatials, sample.N_dim, axis=0)) 
    return singulars, spatials, temporals
    


class PodCoherentStrength(FeatureProcessor):
    def __init__(self, feature_config, folder: Path):
        self.feature_config = feature_config
        self.save_to = folder 
        # Set 
        self._descriptor_function = process_pod_to_1D_Descriptors

    def process_features(self, sample: Sample) -> tuple[Descriptors, TimeSeries]:

        # Specify files
        saved_feature_files = {
            'descriptors':self.save_to / f'{sample.name}_descriptors.csv',
            'timeseries': self.save_to / f'{sample.name}_timeseries.csv'
        }

        # Check if file exists
        if all([ value.exists() for key, value in saved_feature_files.items()]):
            descriptors = read_csv(saved_feature_files['descriptors'])
            timeseries = read_csv(saved_feature_files['timeseries'])
        else:
            # POD
            singulars, spatials, temporals = read_or_process_pod_sample(sample, self.save_to) 

            # Use a helper function to process this 
            descriptors = self.descriptor_function(
                loc_index = sample.loc_index, 
                spatials=spatials, 
                singulars=singulars
                )
            timeseries = temporals
            np.savetxt(saved_feature_files['descriptors'], descriptors, delimiter=",")
            np.savetxt(saved_feature_files['timeseries'], timeseries, delimiter=",")

        return timeseries, descriptors

    @property
    def descriptor_function(self) -> Callable:
        return self._descriptor_function
    

class PodSSIM(FeatureProcessor):
    def __init__(self, feature_config, folder: Path):
        self.feature_config = feature_config
        self.save_to = folder 

    def set_ref_sample(self, ref_sample):
        self.ref_sample = ref_sample # Reference sample

    def process_features(self, sample: Sample) -> tuple[Descriptors, TimeSeries]:
        # Specify files
        saved_feature_files = {
            'descriptors':self.save_to / f'{sample.name}_descriptors.csv',
            'timeseries': self.save_to / f'{sample.name}_timeseries.csv'
        }

        # Check if file exists
        if all([ value.exists() for key, value in saved_feature_files.items()]):
            descriptors = read_csv(saved_feature_files['descriptors'])
            timeseries = read_csv(saved_feature_files['timeseries'])
        else:
            # POD
            singulars, spatials, temporals = read_or_process_pod_sample(sample, self.save_to) 
            singulars_ref, spatials_ref, temporals_ref = read_or_process_pod_sample(self.ref_sample, self.save_to) 

            # Use a helper function to process this 
            descriptors = process_pod_to_1D_Descriptors_SSIM(
                spatials=spatials, 
                spatials_ref=spatials_ref,
                singulars=singulars,
                )
            timeseries = temporals
            np.savetxt(saved_feature_files['descriptors'], descriptors, delimiter=",")
            np.savetxt(saved_feature_files['timeseries'], timeseries, delimiter=",")

        return timeseries, descriptors

    @property
    def descriptor_function(self) -> Callable:
        return self._descriptor_function