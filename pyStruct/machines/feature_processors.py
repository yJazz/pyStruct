from dataclasses import dataclass
from itertools import product

import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from typing import Protocol

from pyStruct.data.dataset import PodModesManager, DmdModesManager
from pyStruct.data.datastructures import *
 

def process_pod_to_1D_Descriptors(sample: PodSample) -> np.ndarray:
    loc_index = sample.loc_index
    spatial_array = np.linalg.norm(sample.pod.X_spatial[:, :, loc_index], axis=1)
    singular = sample.pod.X_s
    spatial = spatial_array
    return np.vstack((singular, spatial)).T

def process_pod_to_timeseries(sample: PodSample) -> np.ndarray:
    return sample.pod.X_temporal


class FeatureProcessor:
    def process_samples(self, samples:list[PodSample]) -> None:
        raise NotImplementedError()


class PodCoherentStrength(FeatureProcessor):
    def __init__(self, feature_config):
        self.feature_config = feature_config
    
    def process_samples(self, sample_set: PodSampleSet) -> None:
        for sample in sample_set.samples:
            descriptors = process_pod_to_1D_Descriptors(sample)
            timeseries = process_pod_to_timeseries(sample)
            sample.set_flowfeatures(timeseries, descriptors)
        return 

    def compose_temporal_matrix(self, samples: list[PodSample]) -> np.ndarray:
        return np.array([sample.pod.X_temporal for sample in samples])
