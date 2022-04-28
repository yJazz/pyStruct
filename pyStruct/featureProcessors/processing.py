import abc

class SpatiotemporalInterface(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self, 
        name: str, 
        N_dim: int, 
        bc: dict,
        X_matrix, 
        dt: float, 
        truncate: int=None
        ):
        raise NotImplementedError

    @abc.abstractmethod
    def _process(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def X_rec(self):
        raise NotImplementedError()


class PodInterface(SpatiotemporalInterface):
    def __init__(
        self, 
        name: str, 
        N_dim: int, 
        bc: dict,
        X_matrix: np.ndarray, 
        dt: float, 
        truncate: int=None
        ):

        self.name = name
        self.N_dim = N_dim
        self._spatials, self._temporals, self._singulars, self.mean = self._process(X_matrix, truncate)

        self.N_t = X_matrix.shape[1]
        self.N_modes = len(self._singulars)
        self.bc = bc
        self.dt = dt
        self._X_rec = self._reconstruct()

    def _process(self, X_matrix):
        raise NotImplementedError()
    
    def _reconstruct(self):
        raise NotImplementedError()

    @property
    def X_rec(self):
        raise NotImplementedError()
    
    @property
    def spatials(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def temporals(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def singulars(self) -> np.ndarray:
        raise NotImplementedError()


class StandardPod(PodInterface):
    def __init__(
        self, 
        name: str, 
        N_dim: int, 
        bc: dict,
        X_matrix: np.ndarray, 
        dt: float, 
        truncate: int=None
        ):
        super(StandardPod, self).__init__(name, N_dim, bc, X_matrix, dt, truncate)

    def _process(self, X_matrix: np.ndarray, truncate: int):
        s, spatial_modes, temporal_coeff, mean_field = standard_pod(X_matrix, truncate)
        print(f"spatial shapes: {spatial_modes.shape}")
        print(f"temporal shapes: {temporal_coeff.shape}")
        print(f'singluars shapes: {s.shape}')

        return spatial_modes, temporal_coeff, s, mean_field

    def _reconstruct(self):
        return self._spatials @ np.diag(self._singulars) @ self._temporals.T
        
    @property
    def X_rec(self):
        return self._X_rec
        


# class PodValidation:
#     """
#     Use this class to check if the pod processing is correct. 
#     Compare the result of : 
#         (1) Reconstruction from POD modes and 
#         (2) Parse snapshots from raw csv tables
#     use `plot_X_rec_fluc` and `plot_X_val_fluc` to visualize.
#     """
#     def __init__(
#         self,
#         spatials: np.ndarray, 
#         temporals:np.ndarray,
#         singulars:np.ndarray,
#         coord:pd.DataFrame,
#         columns: list[str],
#         snapshot_paths: list[str]
#     ):
#         self.spatials = spatials
#         self.temporals = temporals
#         self.singulars = singulars
#         self.coord = coord
#         self.columns = columns
#         self.snapshot_paths = snapshot_paths

#         self.X_rec_fluc = self.get_X_from_reconstruction()
#         self.X_val_mean, self.X_val_fluc = self.get_X_from_snapshots()

#     def compare_rec_and_val(self):
#         return abs(self.X_rec_fluc - self.X_val_fluc) <1E-2 

#     def plot_X_rec_fluc(self, mode=0):
        
#         x = self.coord['X (m)']
#         y = self.coord['Y (m)']
#         z = self.coord['Z (m)']
#         N_grids = len(self.coord)

#         pod_u_fluc = self.X_rec_fluc[0*N_grids: 1*N_grids, mode]
#         pod_v_fluc = self.X_rec_fluc[1*N_grids: 2*N_grids, mode]
#         pod_w_fluc = self.X_rec_fluc[2*N_grids: 3*N_grids, mode]
#         df = pd.DataFrame({
#             'x':x,
#             'y':y,
#             'z':z,
#             'u_0':pod_u_fluc, 
#             'v_0': pod_v_fluc,
#             'w_0':pod_w_fluc
#         })
#         pc = PolarContourPlot(df)
#         pc.plot()
    
#     def plot_X_val_fluc(self, mode=0):
#         # Plot Val
#         val_coord = self.get_coord_from_snapshot()
#         x = val_coord['X (m)']
#         y = val_coord['Y (m)']
#         z = val_coord['Z (m)']
#         N_grids = len(val_coord)

#         val_u_fluc = self.X_val_fluc[0*N_grids: 1*N_grids, mode]
#         val_v_fluc = self.X_val_fluc[1*N_grids: 2*N_grids, mode]
#         val_w_fluc = self.X_val_fluc[2*N_grids: 3*N_grids, mode]
#         val_df = pd.DataFrame({
#             'x':x,
#             'y':y,
#             'z':z,
#             'u_0':val_u_fluc,
#             'v_0':val_v_fluc,
#             'w_0':val_w_fluc,
#         })
#         pc_val = PolarContourPlot(val_df)
#         pc_val.plot()

#     def plot_X_val_mean(self):
#         # Plot Val
#         val_coord = self.get_coord_from_snapshot()
#         x = val_coord['X (m)']
#         y = val_coord['Y (m)']
#         z = val_coord['Z (m)']
#         N_grids = len(val_coord)

#         val_u_mean = self.X_val_mean[0*N_grids: 1*N_grids]
#         val_v_mean = self.X_val_mean[1*N_grids: 2*N_grids]
#         val_w_mean = self.X_val_mean[2*N_grids: 3*N_grids]
#         val_df = pd.DataFrame({
#             'x':x,
#             'y':y,
#             'z':z,
#             'u_mean':val_u_mean,
#             'v_mean':val_v_mean,
#             'w_mean':val_w_mean,
#         })
#         pc_val = PolarContourPlot(val_df)
#         pc_val.plot()


#     def get_X_from_reconstruction(self):
#         X = np.matmul(np.matmul(self.spatials, np.diag(self.singulars)), self.temporals.transpose())
#         self.X_construct = X
#         return X

#     def get_X_from_snapshots(self):
#         print("Getting snapshots...")
#         output =[]
#         for path in tqdm(self.snapshot_paths): 
#             dataset=np.array([])
#             for i, column in enumerate(self.columns):
#                 single_data = self._read_single_file(path, column)
#                 dataset = np.concatenate((dataset, single_data))
#             output.append(dataset)
#         X = np.array(output).transpose()
#         print("Done")
#         X_mean = X.mean(axis=1)
#         X_fluc = X - np.reshape(X_mean, (len(X_mean), 1))
#         return X_mean, X_fluc
    
#     def get_coord_from_snapshot(self):
#         path = self.snapshot_paths[0]
#         df = pd.read_csv(path)
#         return df[['X (m)', 'Y (m)', 'Z (m)']]
    
#     def _read_single_file(self, filepath, column_name):
#         """
#         From file at <filepath>, get the data with the column in <column_name>
#         """
#         data = pd.read_csv(filepath)
#         return data[column_name].values  # numpy


class DmdInterface(SpatiotemporalInterface):
    def _process(self):
        raise NotImplementedError()
    
    @property
    def modes(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def eigenvalues(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def omega(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def amplitudes(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def frequency(self) -> np.ndarray:
        raise NotImplementedError()

    def get_time_response(self, mode):
        raise NotImplementedError()

    def get_mode(self, mode):
        raise NotImplementedError()
        
    def get_mode_time_response(self, mode, t=None):
        raise NotImplementedError()


class Dmd(DmdInterface):
    """ A common class """
    def __init__(
        self, 
        name: str, 
        N_dim: int, 
        bc: dict,
        X_matrix: np.ndarray, 
        dt: float, 
        truncate: int=None
        ):

        self.name = name
        self.N_dim = N_dim
        self._modes, self._eigenvalues, self._amplitudes = self._process(X_matrix, truncate)
        self.N_t = X_matrix.shape[1]
        self.N_modes = len(self._eigenvalues)
        self.bc = bc
        self.dt = dt
        self._X_rec = self._reconstruct()
    
    def _reconstruct(self):
        output = self.get_mode_time_response(mode=0).real
        for i in range(1, self.N_modes-1):
            output += self.get_mode_time_response(mode=i).real
        return output

    @property
    def X_rec(self):
        return self._X_rec
    
    @property
    def modes(self) -> np.ndarray:
        return np.array(np.split(self._modes, self.N_dim, axis=0))

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigenvalues

    @property
    def omega(self) -> np.ndarray:
        """ output dimension"""
        return np.log(self._eigenvalues)/self.dt

    @property
    def amplitudes(self) -> np.ndarray:
        return self._amplitudes

    @property
    def frequency(self) -> np.ndarray:
        return self.omega.imag/(2*np.pi)

    def get_time_response(self, mode):
        t = np.arange(1, self.N_t)
        return self.amplitudes[mode]*np.exp(self.omega[mode]*t)

    def get_mode(self, mode):
        return self.modes[:, mode]
        
    def get_mode_time_response(self, mode, t=None):
        concate =  self._modes[:, mode].reshape(-1, 1) *self.get_time_response(mode).reshape(1, -1)
        if t:
            return np.array(np.split(concate, self.N_dim, axis=0))[:, :, t]
        else:
            return np.array(np.split(concate, self.N_dim, axis=0))


class StandardDmd(Dmd):
    def __init__(
        self, 
        name: str, 
        N_dim: int, 
        bc: dict,
        X_matrix: np.ndarray, 
        dt: float, 
        truncate: int=None
        ):
        super(StandardDmd, self).__init__(name, N_dim, bc, X_matrix, dt, truncate)
    
    def _process(self, X_matrix: np.ndarray, truncate: int) -> tuple[np.ndarray]:
        assert truncate<= X_matrix.shape[1]
        return standard_dmd(X_matrix, truncate)
