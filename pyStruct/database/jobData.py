import abc
import numpy as np


class JobDataInterface(abc.ABC):
    @property
    @abc.abstractmethod
    def bc(self):
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def X_matrix_path(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def coord(self):
        raise NotImplementedError

    @abc.abstractmethod
    def T_wall(self, theta_deg: float):
        raise NotImplementedError
