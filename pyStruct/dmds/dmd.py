import numpy as np
from pyStruct.data.datastructures import BoundaryCondition


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


class Dmd:
    def process(self, X_matrix: np.ndarray):
        raise NotImplementedError()


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

        self.name = name
        self.N_dim = N_dim
        self._modes, self._eigenvalues, self._amplitudes = self._process(X_matrix, truncate)
        self.N_t = X_matrix.shape[1]
        self.N_modes = len(self._eigenvalues)
        self.bc = bc
        self.dt = dt
        self._X_rec = self._reconstruct()
    
    def _process(self, X_matrix: np.ndarray, truncate: int) -> tuple[np.ndarray]:
        assert truncate<= X_matrix.shape[1]
        return standard_dmd(X_matrix, truncate)

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
        