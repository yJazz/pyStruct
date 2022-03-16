from pathlib import Path
import pickle 
from tqdm import tqdm
import numpy as np
import json
import numpy as np
import pandas as pd
from pathlib import Path
from math import ceil
from tqdm import tqdm

from scipy.interpolate  import griddata
import matplotlib.pyplot as plt

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
        

def shuffle(X, y):
    N_samples = X.shape[0]
    assert X.shape[0] == y.shape[0], "The size of X, y pair don't match"

    indices = np.arange(0, N_samples)
    rng = np.random.RandomState(seed=np.random.randint(1E6))
    rng.shuffle(indices)

    return X[indices, ...], y[indices, ...]

def split_train_test(X, y, train=0.8, val=0.2):
    N_samples = X.shape[0]
    assert X.shape[0] == y.shape[0], "The size of X, y pair don't match"

    X_train = X[:int(train * N_samples), ...]
    X_val = X[int(train* N_samples):, ...]
    y_train = y[:int(train * N_samples), ...]
    y_val = y[int(train* N_samples):, ...]
    
    return X_train, y_train, X_val, y_val

def get_strength_of_sink(spatial_mode, coord):
    top_point = coord[coord['Z (m)'] == coord['Z (m)'].max()]
    x_0 = list(top_point['X (m)'])[0]
    y_0 = list(top_point['Y (m)'])[0]
    z_0 = list(top_point['Z (m)'])[0]

    x = coord['X (m)']
    y = coord['Y (m)']
    z = coord['Z (m)']

    norm_of_three_dim = np.linalg.norm(spatial_mode, axis=0)
    distance_array = np.array([(x-x_0), (y-y_0), (z-z_0)])
    distance = np.linalg.norm(distance_array, axis=0)

    return sum(distance * norm_of_three_dim)

    # for i in range(len(coord)):
    #     # (3, N_grid)
    #     norm_of_three_dim = np.linalg.norm([spatial_mode[d, i] for d in range(3) ] )

    #     distance =  np.linalg.norm([(x-x_0), (y-y_0), (z-z_0)])
    #     output += norm_of_three_dim * distance
    # return output
    

def combine_features(X_spatials, X_temporals, X_s, coords):
    """ Strength of sink: The spatial info is treat as the strength of sink wrt distance to the top"""
    N_s, N_m, N_dim, N_grid  = X_spatials.shape

    for sample in tqdm(range(N_s)):
        for mode in range(N_m):
            spatial_mode = X_spatials[sample, mode, :, :]
            spatial_mult = get_strength_of_sink(spatial_mode, coords[sample])
            s_mode = X_s[sample, mode]
            X_temporals[sample, mode, :] = X_temporals[sample, mode, :] * s_mode * spatial_mult
    return X_temporals

    
def find_loc_index(theta_rad, coord):
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
    
def get_training_pairs(workspace, theta_deg, N_t=None):
    """ Get training and """
    workspace = Path(workspace)
    feature_temporal_path = workspace / 'features' / 'X_temporals.pkl'
    feature_spatial_path = workspace / 'features' / 'X_spatials.pkl'
    feature_s_path = workspace / 'features' / 'X_singulars.pkl'
    coords_path = workspace / 'target'/ 'coords.pkl'
    
    X_spatials = read_pickle(feature_spatial_path)
    X_temporals = read_pickle(feature_temporal_path)
    X_s = read_pickle(feature_s_path)
    coords = read_pickle(coords_path)
    N_sample, N_mode, _ = X_temporals.shape


    loc_index =  find_loc_index(theta_rad=theta_deg/180*np.pi, coord=coords[0])
    target_filename = f'target_deg{int(theta_deg)}_loc{loc_index}.pkl'
    target_data_path = workspace / 'target' / target_filename 


    # Weighted by coherent strength
    coherent_strength = X_spatials[:, :, :, loc_index]
    # Rank the Temporal modes by their coherent_strength
    for sample in range(N_sample):
        cs = coherent_strength[sample, :, :]
        sort_key = lambda vec: np.linalg.norm(vec)
        for mode in range(20):
            print(sort_key(cs[mode, :]))
        args = np.apply_along_axis(sort_key, axis=1, arr=cs).argsort()[::-1]
        print(f"sample: {sample}, args:{args}")
        X_temporals[sample, ...] = X_temporals[sample, args, :]


    X = X_temporals

    #  Normalized y: subtract the mean to exclude the temperature level variation 
    # since the 2D crossectional pod doesn't know the length of recirculation region
    y = read_pickle(target_data_path)
    if N_t:
        y = y[:, -N_t:]

    state_point_file = workspace/"signac_statepoint.json"
    with open(state_point_file) as f:
        sp = json.load(f)
    
    bcs = {} 
    for sample_name in sp['aggre_sp']:
        bc = sp['aggre_sp'][sample_name]["cfd_sp"]['bc']
        sample = int(sample_name.split("_")[-1])
        bcs[sample] = bc

    for sample in range(N_sample):
        bc = bcs[bcs[sample]==sample]
        T_c = bc['T_COLD_C']
        T_h = bc['T_HOT_C']
        y[sample, ...] = (y[sample, ...] - T_c)/(T_h)

    y = y - y.mean(axis=1).reshape(-1,1)

    # print("--- Load data ---")
    # print(f"X shape: {X.shape}")
    # print(f"y shape: {y.shape}")

    return X, y


def normalize_the_features(X: np.ndarray) -> np.ndarray:
    # Compute shape 
    N_samples, N_modes, N_t = X.shape
    means = X.mean(axis=-1)
    means = np.reshape(means, (N_samples, N_modes, 1))
    stds = X.std(axis=-1)
    stds = np.reshape(stds, (N_samples, N_modes, 1) )

    X_norm = (X - means)/stds

    return X_norm
    
def reduce_X_dimension(X_slide):
    # keep the last two dimensions
    N_w, N_samples, N_modes, N_t = X_slide.shape
    sub_Xs = [X_slide[i, :, :, :] for i in range(N_w)]
    return np.concatenate(sub_Xs, axis=0)

def reduce_y_dimension(y_slide):
    N_w, N_samples, N_t = y_slide.shape
    sub_ys = [ y_slide[i, :, :] for i in range(N_w)]
    # swap the last axis, in order to keep them intact
    # otherwise reshape will start from the last axis
    return np.concatenate(sub_ys, axis = 0)

    
def sliding_training_pairs(X, y, sequence_length, sequence_stride):
    """ Normalize and sliding """
    # X = normalize_the_features(X)
    
    X_slide = sliding_window(X, sequence_length, sequence_stride)
    y_slide = sliding_window(y,sequence_length, sequence_stride)

    X_reshape = reduce_X_dimension(X_slide)
    y_reshape = reduce_y_dimension(y_slide)

    print(f"----- Slide and Reshape ----- ")
    print(f"reshaped slided X: {X_reshape.shape}")
    print(f"reshaped y: {y_reshape.shape}")

    return X_reshape, y_reshape 
    
def get_data_length(X):
    data_dim = X.ndim
    for d in range(X.ndim -1):
        X = X[-1]
    return len(X)


def sliding_window(X,
                   sequence_length,
                   sequence_stride=1,
                   sampling_rate=1,
                   Y=None,
                   batch_size=128,
                   shuffle=False,
                   seed=None,
                   start_index=None,
                   end_index=None):
    """
    data: ndarray, axis 0 is expected to be the time dimension
    """
    data_length = get_data_length(X)
        
    if start_index:
        if start_index < 0:
            raise ValueError(f'`start_index` must be 0 or greater. Received: '
                             f'start_index={start_index}')
        if start_index >= data_length:
            raise ValueError(f'`start_index` must be lower than the length of the '
                             f'data. Received: start_index={start_index}, for data '
                             f'of length {len(X)}')
    if end_index:
        if start_index and end_index <= start_index:
            raise ValueError(f'`end_index` must be higher than `start_index`. '
                             f'Received: start_index={start_index}, and '
                             f'end_index={end_index} ')
        if end_index >= data_length:
            raise ValueError(f'`end_index` must be lower than the length of the '
                             f'data. Received: end_index={end_index}, for data of '
                             f'length {len(X)}')
        if end_index <= 0:
            raise ValueError('`end_index` must be higher than 0. '
                             f'Received: end_index={end_index}')

    # Validate strides
    if sampling_rate <= 0:
        raise ValueError(f'`sampling_rate` must be higher than 0. Received: '
                         f'sampling_rate={sampling_rate}')
    if sampling_rate >= data_length:
        raise ValueError(f'`sampling_rate` must be lower than the length of the '
                         f'data. Received: sampling_rate={sampling_rate}, for data '
                         f'of length {len(X)}')
    if sequence_stride <= 0:
        raise ValueError(f'`sequence_stride` must be higher than 0. Received: '
                         f'sequence_stride={sequence_stride}')
    if sequence_stride >= data_length:
        raise ValueError(f'`sequence_stride` must be lower than the length of the '
                         f'data. Received: sequence_stride={sequence_stride}, for '
                         f'data of length {len(X)}')

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = data_length
        # end_index = len(X) # assuming axis 0 is the time-step


    # Determine the lowest dtype to store start positions (to lower memory usage).
    num_seqs = end_index - start_index - (sequence_length * sampling_rate) + 1
    if num_seqs < 2147483647:
        index_dtype = 'int32'
    else:
        index_dtype = 'int64'

    # Generate start positions
    start_positions = np.arange(0, num_seqs, sequence_stride, dtype=index_dtype)
    if shuffle:
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(start_positions)

    # For each initial window position, geneate indices (good for treating sampling rate)
    indices = [range(start_positions[i], start_positions[i] + sequence_length * sampling_rate, sampling_rate) for i in
               range(len(start_positions))]

    # from indices, generate sequences
    X_sequences_from_indices = np.array([X[..., ind] for ind in indices])
    return X_sequences_from_indices



class ModesManager:
    """ 
    Organize the propcessed data in a class. 
    Attributes:
        X_spatials: tuples with shape (N_samples, N_modes, N_dims, N_grids)
        X_temporals: tuples with shape (N_samples, N_modes, N_t)
        X_s: tuples with shape (N_samples, N_modes)
        T_walls: dict, key=str(theta_deg), value= tuple with shape (N_samples, N_total)
        bcs: pd.DataFrame

    """
    def __init__(self, name, workspace, to_folder=None):
        self.name = name
        self.workspace = Path(workspace)
        self.X_spatials, self.X_temporals, self.X_s, self.coords, self.T_walls = self._init_data()


        if to_folder:
            self.to_folder = Path(to_folder)
        else:
            self.to_folder = self.workspace / "data"
            self.to_folder.mkdir(parents=True, exist_ok=True)

        # Determine the dimension
        self.N_samples, self.N_modes, self.N_dims, self.N_grids = self.X_spatials.shape
        _, _, self.N_t = self.X_temporals.shape

        # Read the boundary conditions
        self.bcs = self.get_bcs()

        # Determine the plot grids 
        self.ncols = 5
        self.nrows = ceil(self.N_samples / self.ncols)
    
    def _init_data(self):
        """ Initialize the data from the workspace"""
        feature_temporal_path = self.workspace / 'features' / 'X_temporals.pkl'
        feature_spatial_path = self.workspace / 'features' / 'X_spatials.pkl'
        feature_s_path = self.workspace / 'features' / 'X_singulars.pkl'
        coords_path = self.workspace / 'target'/ 'coords.pkl'

        X_spatials = read_pickle(feature_spatial_path)
        X_temporals = read_pickle(feature_temporal_path)
        X_s = read_pickle(feature_s_path)
        coords = read_pickle(coords_path)

        N_sample, N_mode, N_t = X_temporals.shape

        T_walls = {}
        for theta_deg in [0, 5, 10, 15, 20]:
            loc_index =  find_loc_index(theta_rad=theta_deg/180*np.pi, coord=coords[0])
            target_filename = f'target_deg{int(theta_deg)}_loc{loc_index}.pkl'
            target_data_path = self.workspace / 'target' / target_filename 
            y = read_pickle(target_data_path)

            y = y[:, -N_t:]
            state_point_file = self.workspace/"signac_statepoint.json"
            with open(state_point_file) as f:
                sp = json.load(f)
            
            bcs = {} 
            for sample_name in sp['aggre_sp']:
                bc = sp['aggre_sp'][sample_name]["cfd_sp"]['bc']
                sample = int(sample_name.split("_")[-1])
                bcs[sample] = bc

            for sample in range(N_sample):
                bc = bcs[bcs[sample]==sample]
                T_c = bc['T_COLD_C']
                T_h = bc['T_HOT_C']
                y[sample, ...] = (y[sample, ...] - T_c)/(T_h)

            y = y - y.mean(axis=1).reshape(-1,1)
            T_walls[str(theta_deg)] = {'loc_index':loc_index, 'T_wall':y}
        return X_spatials, X_temporals, X_s, coords, T_walls

    def _read_signac(self):
        state_point_file = self.workspace/"signac_statepoint.json"
        with open(state_point_file) as f:
            sp = json.load(f)
        
        bcs = {} 
        for sample_name in sp['aggre_sp']:
            bc = sp['aggre_sp'][sample_name]["cfd_sp"]['bc']
            sample = int(sample_name.split("_")[-1])
            bcs[sample] = bc
        return bcs

    def get_bcs(self):
        """ The property table"""
        bcs = self._read_signac()
        properties = pd.DataFrame()

        for sample in range(self.N_samples):
            m_c = bcs[sample]['M_COLD_KGM3S']
            m_h = bcs[sample]['M_HOT_KGM3S']
            T_c = bcs[sample]['T_COLD_C']
            T_h = bcs[sample]['T_HOT_C']

            d = {
                'sample':sample,
                'm_c': float(m_c),
                'm_h':float(m_h),
                'T_c': float(T_c),
                'T_h':float(T_h),
            }
            properties = pd.concat([properties, pd.DataFrame(d, index=[0])], ignore_index=True)
        return properties

    def plot_spatial_mode(self, mode, dim=0, figsize=(16,12)):
        fig, axes = plt.subplots(self.nrows, self.ncols, figsize=figsize)

        for sample in tqdm(range(self.N_samples)):
            ax = axes[sample//self.ncols, sample%self.ncols]
            ax.set_title(f'sample {sample}')

            y = self.coords[sample]['Y (m)']
            z = self.coords[sample]['Z (m)']
            spatial = self.X_spatials[sample, mode, dim, :]

            ax = self.plot_cartesian(ax, y, z, spatial, t=0, vmin=-0.02, vmax=0.02)

            plt.suptitle(f"Spatial mode {mode}")
            plt.tight_layout()
            plt.savefig(self.to_folder/f'{self.name}_mode_{mode}_spatial.png')
        return

    def plot_temporal_mode(self, mode, N_t=1000, figsize=(16, 12)):
        fig, axes = plt.subplots(self.nrows, self.ncols, figsize=figsize)
        for sample in tqdm(range(self.N_samples)):
            ax = axes[sample//self.ncols, sample%self.ncols]
            ax.set_title(f'sample {sample}')
            if type(N_t) == int:
                ax.plot(self.X_temporals[sample, mode, -N_t:])
            elif hasattr(N_t, '__iter__'):
                ax.plot(self.X_temporals[sample, mode, N_t])
            else:
                raise ValueError("N_t should be int or iterable")
            plt.suptitle(f"Temporal mode {mode}")
            plt.tight_layout()
            plt.savefig(self.to_folder/f'{self.name}_mode_{mode}_temporal.png')
        return

    def plot_T_wall(self, theta_deg, figsize=(16, 12)):
        fig, axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=figsize)

        for sample in range(self.N_samples):
            ax = axes[sample//self.ncols, sample%self.ncols]
            ax.plot(self.T_walls[str(theta_deg)][sample, ...])
            ax.set_title(f"Sample {sample}")
            ax.set_ylim(-0.5, 0.5)
        plt.suptitle(f"T_wall: theta= {theta_deg}")
        plt.tight_layout()
        plt.savefig(self.to_folder/f"T_wall_{theta_deg}.png")
        return

    def plot_cartesian(self, ax, y,z,value, t,resolution = 50,contour_method='cubic', vmin=-0.002, vmax=0.002):
        resolution = str(resolution)+'j'
        X,Y = np.mgrid[min(y):max(y):complex(resolution),   min(z):max(z):complex(resolution)]
        points = [[a,b] for a,b in zip(y,z)]
        Z = griddata(points, value, (X, Y), method=contour_method)

        cs = ax.contourf(X,Y,Z, cmap='jet', vmin=vmin, vmax=vmax)
        return cs