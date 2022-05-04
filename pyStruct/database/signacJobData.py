from pathlib import Path
from typing import Callable
import pandas as pd
import numpy as np

from pyStruct.database.jobData import JobDataInterface
from pyStruct.database.datareaders import read_csv, read_gridfile, read_temperature, find_loc_index

class CfdDataNotExist(Exception):
    def __init__(self, message=None):
        super(CfdDataNotExist, self).__init__(message)

class TimeSeriesDataNotExist(Exception):
    def __init__(self, message=""):
        super(TimeSeriesDataNotExist, self).__init__(message)

class SnapshotMatrixFileNotFoundError(Exception):
    ...


class SignacJobData(JobDataInterface):
    def __init__(self, job, parent_job, bc_mapper=None):
        self.job = job
        self.parent_job = parent_job
        self.bc_mapper = bc_mapper

        path = Path(self.parent_job.workspace()) / 'cfd'
        if not path.exists():
            raise CfdDataNotExist()

    def __repr__(self):
        return f'job: {self.job}, parent_job: {self.parent_job}'

    @property
    def bc(self):
        # read sp
        if self.bc_mapper:
            return {self.bc_mapper[key]: value for key,value in self.job.sp.cfd_sp.bc.items() }
        else:
            return {key: value for key,value in self.job.sp.cfd_sp.bc.items() }

    @property
    def X_matrix_path(self) -> np.ndarray:
        path = Path(self.job.workspace()) /'X_matrix.csv'
        if not path.exists():
            raise SnapshotMatrixFileNotFoundError(f"{path}")
        # array = read_csv(path)
        return path

    @property
    def coord(self) -> pd.DataFrame:
        path = Path(self.parent_job.workspace()) /"TimeSeries" / 'coord.csv'
        if not path.exists(): 
            raise TimeSeriesDataNotExist(f"{path}")
        return read_gridfile(path)

    def T_wall(self, theta_deg: float, normalize=False) -> np.ndarray:
        theta_rad = theta_deg / 180 * np.pi
        loc = find_loc_index(theta_rad, self.coord)
        path = Path(self.parent_job.workspace()) / "TimeSeries" / f'loc_{loc}.csv'
        if not path.exists():
            raise TimeSeriesDataNotExist(f'{path}')
        tempearture = read_temperature(path) - 273.15

        if normalize:
            T_c = self.bc['T_c']
            T_h = self.bc['T_h']
            return (tempearture-T_c)/(T_h - T_c)
        else:
            return tempearture


