import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from typing import Protocol

from pyStruct.data.dataset import ModesManager


class FeatureProcessor(Protocol):
    """ This class defines the protocol of the feature processor 
        If the object has get_X(), and get_y(), it is a feature processor
    """

    def get_X(self):
        pass

    def get_y(self):
        pass

    def create_featrue_table(self):
        pass


class CoherentStrength:
    """ 
    The pod modes' impact on T_probe is ranked by the
    1. The global energy of the mode (singular value)
    2. The strength of the coherent structure (spatiral )
    """
    def __init__(self, config):
        self.config = config
        self.pods = ModesManager(name='', workspace=config['workspace'], to_folder=None)
        # self.features = ['sample', 'mode', 'theta_deg', 'singualr', 'probe_strength']
        self.feature_table = self._create_feature_table(config['theta_deg'])
        self._rank_the_modes_by_strength()
    
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
            'probe_strength': probe_strength_flat
            })

    def _rank_the_modes_by_strength(self):
        self.feature_table['rank'] = np.zeros(len(self.feature_table))
        _, useful_modes_idx = self.get_X()

        for sample in range(self.pods.N_samples):
            feature_by_sample = self.feature_table[self.feature_table['sample'] == sample]
            sorted_feature = feature_by_sample.sort_values(by=['probe_strength'], ascending=True)
            ids = sorted_feature.index
            for rank, id in enumerate(ids):
                self.feature_table.loc[id, 'rank'] = rank
        return
            
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
            feature_per_sample = self.feature_table[self.feature_table['sample'] == sample]
            ids = feature_per_sample.index

            # Sort the dataframe by the strength
            sorted_features_per_sample = feature_per_sample.sort_values(by=['probe_strength'], ascending=False)
            args = sorted_features_per_sample['mode'].values
            cs = sorted_features_per_sample['probe_strength'].values
            for mode, arg in enumerate(args[:self.config['N_modes']]):
                X[sample, mode, :] = X_temporals[sample, arg, :] * cs[arg] *10
            useful_modes_idx.append(args)

        return X, useful_modes_idx
        
    def get_y(self):
        y = self.pods.T_walls[str(self.config['theta_deg'])]['T_wall']
        return y


 
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



def get_temporal_behavior(temporal_mode):
    """ Reflect the freqeuency of the mode"""
    freq, psd = get_psd(0.005, temporal_mode)
    sum = psd[:20].sum() *1E4
    return sum


    


    


