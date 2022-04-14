from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


ProcessingFunction = Callable[[np.ndarray, int], tuple]

def processing_pod(x: np.array, truncate: int = None) -> tuple:
    x_mean = x.mean(axis=1)
    x_podmx = x - np.reshape(x_mean, (len(x_mean), 1))
    x_podmx = np.array(x_podmx)
    print("Get Singular & eigenvectors...")
    # SVD: U =  u * np.diag(s) * v
    u, s, v_h = np.linalg.svd(x_podmx, full_matrices=False)
    v = v_h.transpose().conjugate()
    #     s = np.diag(s)[:truncate, :truncate]
    u = u[:, :truncate]
    v = v[:, :truncate]

    # Get np from cp
    # u = u.get()  # spatial modes
    # s = s.get()  # singular values
    # v = v.get()  # temporal modes

    spatial_modes = u
    temporal_coeff = v
    print("Done")
    return s, spatial_modes, temporal_coeff

def processing_dmd(x: np.ndarray, truncate: int = None) -> tuple:
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
    b = np.linalg.pinv(Phi) @x1
    amplitudes = b

    return modes, eigenvalues, amplitudes



class PodProcessor:
    def __init__(
        self, 
        columns: list[str], 
        N_t: int, 
        gridfile: Path,
        time_series_folder: Path, 
        dst_folder: Path
        ):

        self.columns = columns
        self.N_t = N_t
        self.gridfile = gridfile
        self.time_series_folder = time_series_folder
        self.dst_folder = dst_folder
        self.dst_folder.mkdir(parents=True, exist_ok=True)
        
        coord = pd.read_csv(self.gridfile)
        N_grid = len(coord)
        self.N_grid = N_grid

    def _get_time_series(self, file_path: Path):
        file = pd.read_csv(file_path)
        return [tuple(file[column].values[-self.N_t:]) for column in self.columns]

    def get_input_matrix(self): 
        print("Read time histroy")
        N_dim = len(self.columns)
        place_holder = [[] for i in range(N_dim)]

        for loc in range(self.N_grid):
            # Extract the time-series columns from each loc file 
            file_path = self.time_series_folder / f"loc_{loc}.csv"
            data = self._get_time_series(file_path) # a nested list, len(data) = number of columns
            # 
            for d in range(N_dim):
                place_holder[d].append(data[d])
            
        x_matrix = np.concatenate([np.array(place_holder[i]) for i in range(N_dim)], axis=0)
        return x_matrix

    def process(self, truncate:int=None):
        x_matrix = self.get_input_matrix()
        s, spatials, temporals = processing_pod(x_matrix, truncate)

        # save
        s_file = self.dst_folder / 's.csv'
        spatials_file = self.dst_folder / 'spatials.csv'
        temporals_file = self.dst_folder / 'temporals.csv'
        coord_file = self.dst_folder / 'coord.csv'

        np.savetxt(s_file, s, delimiter=',')
        np.savetxt(spatials_file, spatials, delimiter=',')
        np.savetxt(temporals_file, temporals, delimiter=',')
        shutil.copy(self.gridfile, coord_file)

        return s, spatials, temporals

    def collect_spatial(self):
        spatials = np.genfromtxt(self.dst_folder / 'spatials.csv', delimiter=',')
        return np.array([ split_modes_by_dimension(spatials, i, self.N_grids) for i in range(self.N_modes)])

    def collect_temporal(self):
        temporals = np.genfromtxt(self.dst_folder / 'temporals.csv', delimiter=',')
        return np.array([get_temporal_pod(temporals, mode) for mode in range(self.N_modes)])

    def collect_singular_value(self):
        s = np.genfromtxt(self.dst_folder / 's.csv', delimiter=',')
        return np.array([s[mode] for mode in range(self.N_modes)]) 


class DmdProcessor(PodProcessor):
    def __init__(
        self, 
        columns: list[str], 
        N_t: int, 
        gridfile: Path,
        time_series_folder: Path, 
        dst_folder: Path
        ):

        self.columns = columns
        self.N_t = N_t
        self.gridfile = gridfile
        self.time_series_folder = time_series_folder
        self.dst_folder = dst_folder
        self.dst_folder.mkdir(parents=True, exist_ok=True)
        
        coord = pd.read_csv(self.gridfile)
        N_grid = len(coord)
        self.N_grid = N_grid

    def process(self, truncate:int=None):
        x_matrix = self.get_input_matrix()
        modes, eigenvalues, amplitudes = processing_dmd(x_matrix, truncate)

        # save
        modes_file = self.dst_folder/'modes.csv'
        eigenvalues_file = self.dst_folder / 'eigenvalues.csv'
        amplitudes_file = self.dst_folder / 'amplitudes.csv'
        coord_file = self.dst_folder / 'coord.csv'

        np.savetxt(modes_file, modes, delimiter=',')
        np.savetxt(eigenvalues_file, eigenvalues, delimiter=',')
        np.savetxt(amplitudes_file, amplitudes, delimiter=',')
        shutil.copy(self.gridfile, coord_file)

        return modes, eigenvalues, amplitudes




def split_modes_by_dimension(spatial_modes, mode_id, n_grids):
    """ Parse the POD modes 
        INPUT
        --------------------
        spatial_modes: array-like, with shape: (n_dim*n_grid, n_modes), the content is from the saved csv file 
        mode: int, the id of the mode 
    """
    output =[]
    dim = int(spatial_modes.shape[0] / n_grids)
    for d in range(dim):
        output.append(spatial_modes[:, mode_id][n_grids * d: n_grids * (d + 1)])
    return np.array(output)


def get_temporal_pod(temporal_coeff, mode):
    return temporal_coeff[:, mode]


def get_spatial_dmd(dmd_modes, mode_id, n_grids):
    output = []

    




class PolarContourPlot:
    def __init__(
        self, 
        df: pd.DataFrame,
        cmap = 'jet', 
    ):
        assert type(df) == pd.DataFrame
        assert list(df.columns[:3]) == ['x', 'y', 'z']
        self.df = df
        self.cmap = cmap
        # 
        self.columns = self.df.columns[3:]
        self.N_grids = len(df)
        self.R_resolution = 10
        self.theta_resolution = 20
        self.level = 20
    

    def plot(self, cmap='jet', title='', figsize=None, show=True):
        if not figsize:
            figsize = (4*len(self.columns), 4 )
        fig, axes = plt.subplots(
            nrows=1, ncols=len(self.columns), 
            subplot_kw={'projection': 'polar'},
            figsize=figsize
            )
        self.radius = self.df['z'].max()

        if len(self.columns) == 1:
            ax = axes
            ax = self._plot_single(fig, ax, self.columns[0])
        else:
            for i, col in enumerate(self.columns):
                ax = axes[i]
                ax = self._plot_single(fig, ax, col)
        
        fig.suptitle(title, fontweight='semibold', y=1.1)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax


    def _plot_single(self, fig, ax, col):
        # Plot init
        ax.set_title(col)
        ax.set_rticks(self.radius * 0.1 * np.arange(2, 12, 2))
        ax.set_yticklabels(['' for i in range(5)])
        ax.set_xticks(ax.get_xticks().tolist()) 
        ax.set_xticklabels(['$90^o$', '$45^o$', '$0^o$', '$315^o$', '$270^o$', '$225^o$', '$180^o$', '$135^o$'])

        # Get data
        Theta, R, output_matrix = self._map_data(col)
        # Plot
        cs = ax.contourf(Theta,
                         R,
                         output_matrix,
                         self.level,
                         cmap=self.cmap,
                         alpha=0.8)

        # Color bar
        cbar = fig.colorbar(cs, ax=ax, orientation="horizontal", format="%.2f")
        cbar.ax.locator_params(nbins=7)
        return fig, ax, cs

    def set_resolution(self, R_resolution, theta_resolution, level):
        self.R_reolution = R_resolution
        self.theta_resolution = theta_resolution
        self.level = level

    def _find_closest_point(self, file, x, y, z):
        x_table = file['x']
        y_table = file['y']
        z_table = file['z']
        distance = np.sqrt((x - x_table) ** 2 + (y - y_table) ** 2 + (z - z_table) ** 2)
        file['distance'] = distance
        representative_point = file['distance'].argmin()
        representative_point_col = file.iloc[representative_point]
        return representative_point_col

    def _map_data(self, column_name):
        R = np.linspace(0, self.radius, self.R_resolution)
        Theta = np.linspace(0, 2 * np.pi, self.theta_resolution)
        output_matrix = np.zeros((self.R_resolution, self.theta_resolution))

        x = self.df['x']
        for j, theta in enumerate(Theta):
            for i, r in enumerate(R):
                y = -r * np.cos(theta)
                z = r * np.sin(theta)
                col = self._find_closest_point(self.df, x,  y, z)
                output_matrix[i][j] = col[column_name]
        return Theta, R, output_matrix

        



class PodValidation:
    """
    Use this class to check if the pod processing is correct. 
    Compare the result of : 
        (1) Reconstruction from POD modes and 
        (2) Parse snapshots from raw csv tables
    use `plot_X_rec_fluc` and `plot_X_val_fluc` to visualize.
    """
    def __init__(
        self,
        spatials: np.ndarray, 
        temporals:np.ndarray,
        singulars:np.ndarray,
        coord:pd.DataFrame,
        columns: list[str],
        snapshot_paths: list[str]
    ):
        self.spatials = spatials
        self.temporals = temporals
        self.singulars = singulars
        self.coord = coord
        self.columns = columns
        self.snapshot_paths = snapshot_paths

        self.X_rec_fluc = self.get_X_from_reconstruction()
        self.X_val_mean, self.X_val_fluc = self.get_X_from_snapshots()

    def compare_rec_and_val(self):
        return abs(self.X_rec_fluc - self.X_val_fluc) <1E-2 

    def plot_X_rec_fluc(self, mode=0):
        
        x = self.coord['X (m)']
        y = self.coord['Y (m)']
        z = self.coord['Z (m)']
        N_grids = len(self.coord)

        pod_u_fluc = self.X_rec_fluc[0*N_grids: 1*N_grids, mode]
        pod_v_fluc = self.X_rec_fluc[1*N_grids: 2*N_grids, mode]
        pod_w_fluc = self.X_rec_fluc[2*N_grids: 3*N_grids, mode]
        df = pd.DataFrame({
            'x':x,
            'y':y,
            'z':z,
            'u_0':pod_u_fluc, 
            'v_0': pod_v_fluc,
            'w_0':pod_w_fluc
        })
        pc = PolarContourPlot(df)
        pc.plot()
    
    def plot_X_val_fluc(self, mode=0):
        # Plot Val
        val_coord = self.get_coord_from_snapshot()
        x = val_coord['X (m)']
        y = val_coord['Y (m)']
        z = val_coord['Z (m)']
        N_grids = len(val_coord)

        val_u_fluc = self.X_val_fluc[0*N_grids: 1*N_grids, mode]
        val_v_fluc = self.X_val_fluc[1*N_grids: 2*N_grids, mode]
        val_w_fluc = self.X_val_fluc[2*N_grids: 3*N_grids, mode]
        val_df = pd.DataFrame({
            'x':x,
            'y':y,
            'z':z,
            'u_0':val_u_fluc,
            'v_0':val_v_fluc,
            'w_0':val_w_fluc,
        })
        pc_val = PolarContourPlot(val_df)
        pc_val.plot()

    def plot_X_val_mean(self):
        # Plot Val
        val_coord = self.get_coord_from_snapshot()
        x = val_coord['X (m)']
        y = val_coord['Y (m)']
        z = val_coord['Z (m)']
        N_grids = len(val_coord)

        val_u_mean = self.X_val_mean[0*N_grids: 1*N_grids]
        val_v_mean = self.X_val_mean[1*N_grids: 2*N_grids]
        val_w_mean = self.X_val_mean[2*N_grids: 3*N_grids]
        val_df = pd.DataFrame({
            'x':x,
            'y':y,
            'z':z,
            'u_mean':val_u_mean,
            'v_mean':val_v_mean,
            'w_mean':val_w_mean,
        })
        pc_val = PolarContourPlot(val_df)
        pc_val.plot()


    def get_X_from_reconstruction(self):
        X = np.matmul(np.matmul(self.spatials, np.diag(self.singulars)), self.temporals.transpose())
        self.X_construct = X
        return X

    def get_X_from_snapshots(self):
        print("Getting snapshots...")
        output =[]
        for path in tqdm(self.snapshot_paths): 
            dataset=np.array([])
            for i, column in enumerate(self.columns):
                single_data = self._read_single_file(path, column)
                dataset = np.concatenate((dataset, single_data))
            output.append(dataset)
        X = np.array(output).transpose()
        print("Done")
        X_mean = X.mean(axis=1)
        X_fluc = X - np.reshape(X_mean, (len(X_mean), 1))
        return X_mean, X_fluc
    
    def get_coord_from_snapshot(self):
        path = self.snapshot_paths[0]
        df = pd.read_csv(path)
        return df[['X (m)', 'Y (m)', 'Z (m)']]
    
    def _read_single_file(self, filepath, column_name):
        """
        From file at <filepath>, get the data with the column in <column_name>
        """
        data = pd.read_csv(filepath)
        return data[column_name].values  # numpy

