from dataclasses import dataclass
from itertools import product

import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from typing import Protocol

from pyStruct.data.dataset import PodModesManager, DmdModesManager
from pyStruct.machines.datastructures import *
 
@dataclass
class PodModes1DDescriptors:
    singular: np.ndarray
    spatial: np.ndarray
    # temporal: np.ndarray

def process_pod_to_1D_Descriptors(sample: Sample, theta_deg: float):
    loc_index = sample.walls[f'{theta_deg:.2f}'].loc_index
    spatial_array = np.linalg.norm(sample.pod.X_spatial[:, :, loc_index], axis=1)
    return PodModes1DDescriptors(
        singular = sample.pod.X_s,
        spatial = spatial_array
        # temporals = 
    )

def get_temporal_behavior(temporal_mode):
    """ Reflect the freqeuency of the mode"""
    freq, psd = get_psd(0.005, temporal_mode)
    sum = psd[:20].sum() *1E4
    return sum

def get_psd(dt_s, x):
    """
    Get fft of the samples
    :param dt_s: time step
    :param x: samples
    :return:
    """

    # 2-side FFT
    N = len(x)
    xdft = fft(x)
    freq = fftfreq(N, dt_s)

    # convert 2-side to 1-side
    if N % 2 == 0:
        xdft_oneside = xdft[0:int(N / 2 )]
        freq_oneside = freq[0:int(N / 2 )]
    else:
        xdft_oneside = xdft[0:int((N - 1) / 2)+1]
        freq_oneside = freq[0:int((N - 1) / 2)+1]


    # Power spectrum
    Fs = 1 / dt_s
    psdx = 1 / (Fs * N) * abs(xdft_oneside)**2
    psdx[1:-1] = 2 * psdx[1:-1] # power for one-side
    return freq_oneside, psdx

def get_pods_and_samples(workspace, theta_degs, normalize_y):
    """ A messy code... should refactor later"""
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

        # Crate a data structure
        bc = BoundaryCondition(m_c, m_h, T_c, T_h)
        pod = PodFeatures(coord=coord, X_spatial=X_spatial, X_temporal=X_temporal, X_s=X_s)

        walls = {}
        for theta_deg in theta_degs:
            T_wall = pod_manager.T_walls[f'{theta_deg:.2f}']['T_wall'][sample, :]
            loc_index =pod_manager.T_walls[f'{theta_deg:.2f}']['loc_index']
            wall = Wall(theta_deg=theta_deg, loc_index=loc_index, temperature=T_wall)
            walls[f'{theta_deg:.2f}'] = wall
        samples.append(Sample(name=f's{sample}', bc=bc, pod=pod, walls=walls))
    return samples


def get_probe_coord(theta, coord):
    x = coord.loc[0, 'X (m)']
    r = coord['Z (m)'].max()
    y = r * np.sin(theta)
    z = r * np.cos(theta)

    x_coord = coord['X (m)']
    y_coord = coord['Y (m)']
    z_coord = coord['Z (m)']
    distance = np.linalg.norm((x-x_coord, y-y_coord, z-z_coord), axis=0)
    coord['distance'] = distance
    index = coord[coord['distance'] == coord['distance'].min()].index[0]
    return index

def get_spatial_strength(spatial_mode, coord, theta_in_rad):
    N_dim = spatial_mode.shape[0]

    x = coord['X (m)']
    y = coord['Y (m)']
    z = coord['Z (m)']

    if N_dim==3:
        u, v, w = spatial_mode
        df= pd.DataFrame(
            {
                'x':x, 
                'y':y, 
                'z':z, 
                'u':u, 
                'v':v, 
                'w':w, 
            }
        )
        probe_id = get_probe_coord(theta_in_rad, coord)
        x_probe = df.loc[probe_id, 'x']
        y_probe = df.loc[probe_id, 'y']
        z_probe = df.loc[probe_id, 'z']

        df['distance'] = np.linalg.norm((x-x_probe, y-y_probe, z-z_probe), axis=0)
        df['u_value'] = df['distance']* df['u']
        df['v_value'] = df['distance']* df['v']
        df['w_value'] = df['distance']* df['w']
        
        return np.linalg.norm(df[['u_value', 'v_value', 'w_value']])
    else:
        u = spatial_mode[0, :]
        df= pd.DataFrame(
            {
                'x':x, 
                'y':y, 
                'z':z, 
                'u':u, 
            }
        )
        probe_id = get_probe_coord(theta_in_rad, coord)
        x_probe = df.loc[probe_id, 'x']
        y_probe = df.loc[probe_id, 'y']
        z_probe = df.loc[probe_id, 'z']

        df['distance'] = np.linalg.norm((x-x_probe, y-y_probe, z-z_probe), axis=0)
        df['u_value'] = df['distance']* df['u']
        
        return np.linalg.norm(df[['u_value']])


class FeatureProcessor:
    def _collect_samples(self, inputs: list[dict]) -> None:
        raise NotImplementedError()

    def get_structure_tables(self) -> pd.DataFrame:
        """ 
        Crate structure tables for structure predictors
        """
        raise NotImplementedError()

    def compose_temporal_matrix(self, ) -> np.ndarray:
        raise NotImplementedError()

    def get_temporal_signals_outputs(self, inputs: dict) -> np.ndarray:
        raise NotImplementedError()

    def reconstruct(self, temporal_matrix, weights) -> np.ndarray:
        raise NotImplementedError()


class PodCoherentStrength(FeatureProcessor):
    def __init__(self, feature_config):
        self.feature_config = feature_config
        self.samples = self._collect_samples()
        self.training_samples = self.samples[:self.feature_config.N_trains]
        self.testing_samples = self.samples[self.feature_config.N_trains:]


    def _collect_samples(self) ->  list[Sample]:
        # Dependy inversion: the domain model doesn't need to know 
        # how data is loaded
        workspace = self.feature_config.workspace
        normalize_y = self.feature_config.normalize_y
        theta_degs = self.feature_config.theta_degs

        # Get data
        samples = get_pods_and_samples(workspace, theta_degs, normalize_y)
        return samples

    def get_structure_tables(self) -> pd.DataFrame:
        """ 
        The pod modes are converted into series of pod features
        """
        # search for the case that mathces input
        N_modes = self.feature_config.N_modes
        features_placeholder=[]

        for sample, theta_deg in product(self.training_samples, self.feature_config.theta_degs):
            # Convert mode info to descriptors
            descriptors = process_pod_to_1D_Descriptors(sample, theta_deg)

            # Create features
            features = pd.DataFrame({
                'mode': np.arange(N_modes),
                'spatial': descriptors.spatial,
                'singular': descriptors.singular
            })
            # Get bc 
            bc_names = sample.bc.__annotations__.keys()
            for bc_name in bc_names:
                features[bc_name] = np.ones(N_modes) * sample.bc[bc_name]
            features_placeholder.append(features)
        
        # Create dataframe
        structure_table = pd.concat(features_placeholder, ignore_index=True)
        return structure_table

    def compose_temporal_matrix(self, samples: list[Sample]) -> np.ndarray:
        return np.array([sample.pod.X_temporal for sample in samples])



class CoherentStrength_drep(FeatureProcessor):
    """ 
    The pod modes' impact on T_probe is ranked by the
    1. The global energy of the mode (singular value)
    2. The strength of the coherent structure (spatiral )
    """
    def __init__(self, config):
        self.config = config
        self.pods = PodModesManager(name='', workspace=config['workspace'],normalize_y=config['normalize_y'], to_folder=None)

        # Create feature table
        self.feature_table = self._create_feature_table()
        # Rank feature table
        self._rank_the_modes_by_strength()
    
    def _create_feature_table(self) -> pd.DataFrame:
        print("Processing feature table")
        feature_table_all_theatas = [self._create_feature_theta_table(theta_deg) for theta_deg in self.config['theta_degs']]
        feature_table = pd.concat(feature_table_all_theatas, ignore_index=True)
        return feature_table

    def _create_feature_theta_table(self, theta_deg):
        # Read statepoint
        bcs = self.pods.bcs
        X_temporals = self.pods.X_temporals
        X_s = self.pods.X_s
        X_spatials = self.pods.X_spatials
        N_sample, N_mode, N_dim, _ = X_spatials.shape
        
        loc_index = self.pods.T_walls[str(theta_deg)]['loc_index']
        X_probe = X_spatials[:, :, :, loc_index]

        convert_to_norm = lambda vec: np.linalg.norm(vec) 
        X_probe_norm = np.apply_along_axis(convert_to_norm, axis=2, arr=X_probe)


        singulars_flat = [X_s[sample, mode] for sample in range(N_sample) for mode in range(N_mode)]
        probe_strength_flat = [X_probe_norm[sample, mode] for sample in range(N_sample) for mode in range(N_mode)]
        
        mc_flat = [bcs[bcs['sample'] == sample]['m_c'].iloc[0] for sample in range(N_sample) for mode in range(N_mode)]
        mh_flat = [bcs[bcs['sample']==sample]['m_h'].iloc[0] for sample in range(N_sample) for mode in range(N_mode)]
        vel_ratio = np.array(mc_flat)/np.array(mh_flat)
        Tc_flat = [bcs[bcs['sample']==sample]['T_c'].iloc[0] for sample in range(N_sample) for mode in range(N_mode)]
        Th_flat = [bcs[bcs['sample']==sample]['T_h'].iloc[0] for sample in range(N_sample) for mode in range(N_mode)]

        df =  pd.DataFrame({
            'sample': [sample for sample in range(N_sample) for mode in range(N_mode)],
            'mode': [mode for sample in range(N_sample) for mode in range(N_mode)],
            'theta_deg':theta_deg,
            'm_c': mc_flat,
            'm_h':mh_flat,
            'vel_ratio': vel_ratio,
            'T_c':Tc_flat,
            'T_h':Th_flat,
            'singular': singulars_flat,
            'probe_strength': probe_strength_flat,
            })
        for i in range(N_dim):
            df[f'probe_phase_{i}'] = [X_probe[sample, mode, i] for sample in range(N_sample) for mode in range(N_mode)]
        return df

    def _rank_the_modes_by_strength(self):
        print("Rank feature table")
        self.feature_table['rank'] = np.zeros(len(self.feature_table))
        for theta_deg in self.config['theta_degs']:
            for sample in range(self.pods.N_samples):
                feature_by_sample = self.feature_table[self.feature_table['sample'] == sample]
                feature_by_sample_theta = feature_by_sample[feature_by_sample['theta_deg'] == theta_deg]
                sorted_feature = feature_by_sample_theta.sort_values(by=['probe_strength'], ascending=False)
                ids = sorted_feature.index
                for rank, id in enumerate(ids):
                    self.feature_table.loc[id, 'rank'] = int(rank)
        self.feature_table.sort_values(by=['theta_deg', 'sample', 'rank'], ignore_index=True, inplace=True)
        return
            
    def get_bcs(self):
        return self.pods.bcs

    def get_training_pairs(self, theta_deg: float):
        X, _ = self.get_X()
        y = self.get_y(theta_deg)
        return X, y

    def get_X(self):

        X_temporals = self.pods.X_temporals
        X = np.zeros((self.pods.N_samples, self.config['N_modes'], self.pods.N_t))
        useful_modes_idx = []


        for sample in range(self.pods.N_samples):
            # Filter the sample
            feature_per_sample = self.feature_table[self.feature_table['sample'] == sample].sort_values(by=['rank'])

            # Sort the dataframe by the strength
            ranked_modes = feature_per_sample['mode'].values
            cs = feature_per_sample['probe_strength'].values
            # cs = feature_per_sample['probe_phase_3'].values

            for i, rank in enumerate(ranked_modes[:self.config['N_modes']]):
                X[sample, i, :] = X_temporals[sample, rank, :] * cs[rank] *10
                # X[sample, i, :] = X_temporals[sample, rank, :] 

            useful_modes_idx.append(ranked_modes)

        return X, useful_modes_idx
        
    def get_y(self, theta_deg: float):
        y = self.pods.T_walls[str(theta_deg)]['T_wall']
        return y



class SimpleTemporal:
    def __init__(self, config):
        self.config = config
        self.pods = ModesManager(name='', workspace=config['workspace'], to_folder=None)
        # self.features = ['sample', 'mode', 'theta_deg', 'singualr', 'probe_strength']
        self.feature_table = self._create_feature_table(config['theta_deg'])
        self._rank()
    def _rank(self):
        self.feature_table['rank'] = self.feature_table['mode']
    
    def _create_feature_table(self, theta_deg):
        # Read statepoint
        bcs = self.pods.bcs


        X_temporals = self.pods.X_temporals
        X_s = self.pods.X_s
        X_spatials = self.pods.X_spatials

        N_sample, N_mode, _, _ = X_spatials.shape
        
        loc_index = self.pods.T_walls[str(theta_deg)]['loc_index']
        X_probe = X_spatials[:, :, :, loc_index]

        convert_to_norm = lambda vec: np.linalg.norm(vec) 
        X_probe_norm = np.apply_along_axis(convert_to_norm, axis=2, arr=X_probe)


        singulars_flat = [X_s[sample, mode] for sample in range(N_sample) for mode in range(N_mode)]
        probe_strength_flat = [X_probe_norm[sample, mode] for sample in range(N_sample) for mode in range(N_mode)]
        probe_phase = [X_probe[sample, mode, 0] for sample in range(N_sample) for mode in range(N_mode)]

        mc_flat = [bcs[bcs['sample'] == sample]['m_c'].iloc[0] for sample in range(N_sample) for mode in range(N_mode)]
        mh_flat = [bcs[bcs['sample']==sample]['m_h'].iloc[0] for sample in range(N_sample) for mode in range(N_mode)]
        vel_ratio = np.array(mc_flat)/np.array(mh_flat)
        Tc_flat = [bcs[bcs['sample']==sample]['T_c'].iloc[0] for sample in range(N_sample) for mode in range(N_mode)]
        Th_flat = [bcs[bcs['sample']==sample]['T_h'].iloc[0] for sample in range(N_sample) for mode in range(N_mode)]


        return pd.DataFrame({
            'sample': [sample for sample in range(N_sample) for mode in range(N_mode)],
            'mode': [mode for sample in range(N_sample) for mode in range(N_mode)],
            'm_c': mc_flat,
            'm_h':mh_flat,
            'vel_ratio': vel_ratio,
            'T_c':Tc_flat,
            'T_h':Th_flat,
            'singular': singulars_flat,
            'probe_strength': probe_strength_flat,
            'probe_phase': probe_phase
            })

    def get_bcs(self):
        return self.pods.bcs

    def get_training_pairs(self):
        X, _ = self.get_X()
        y = self.get_y()
        return X, y

    def get_X(self):

        X_temporals = self.pods.X_temporals
        X = np.zeros((self.pods.N_samples, self.config['N_modes'], self.pods.N_t))
        useful_modes_idx = []


        for sample in range(self.pods.N_samples):
            # Filter the sample
            feature_per_sample = self.feature_table[self.feature_table['sample'] == sample].sort_values(by=['rank'])

            for mode in range(self.config['N_modes']):
                X[sample, mode, :] = X_temporals[sample, mode, :]

        return X, None
        
    def get_y(self):
        y = self.pods.T_walls[str(self.config['theta_deg'])]['T_wall']
        return y

class DmdFeatures:
    def __init__(self, config):
        self.dmds =DmdModesManager(
            name='', 
            workspace=config['workspace'],
            N_t = config['N_t'],
            normalize_y=config['normalize_y'],
            to_folder=None
            ) 
        pass


    def create_feature_table(self):
        pass

    def get_training_pairs(self, theta_deg:float):
        pass

    def get_X(self):
        X_modes = self.dmds.X_modes
        return X_modes

    def get_y(self):

        pass