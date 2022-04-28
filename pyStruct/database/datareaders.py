import pickle
import pandas as pd
import numpy as np

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_csv(path):
    return pd.read_csv(path, header=None).to_numpy()

def read_temperature(path):
    return pd.read_csv(path)['Temperature (K)'].values

def read_gridfile(path):
    return pd.read_csv(path)

def find_loc_index(theta_deg: float, coord: pd.DataFrame):
    theta_rad = theta_deg/180 * 2* np.pi
    assert abs(theta_rad) <= 2*np.pi, "check the theta_rad: should be in rad"
    x = 0.0635
    r = coord['Z (m)'].max()
    y = r * np.sin(theta_rad)
    z = r * np.cos(theta_rad)

    x_coord = coord['X (m)']
    y_coord = coord['Y (m)']
    z_coord = coord['Z (m)']
    distance = np.linalg.norm((x-x_coord, y-y_coord, z-z_coord), axis=0)
    coord['distance'] = distance
    index = coord[coord['distance'] == coord['distance'].min()].index[0]
    return index