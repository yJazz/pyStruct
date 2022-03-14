import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq


 
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