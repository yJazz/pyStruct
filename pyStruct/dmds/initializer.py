from pathlib import Path
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import signac
from pyStruct.data.selector import DatabaseSelector, get_parent_job
from pyStruct.data.selector import DatabaseSelector, get_parent_job
from pyStruct.data.dataset import find_loc_index
from typing import Callable



def initiate_jobs_from_database(
    proc_database_workspace: str, 
    functions: list[Callable]
    ):
    db = DatabaseSelector(proc_database_workspace)
    jobs = db.filter(*functions)
    return jobs

def read_csv(path):
    return pd.read_csv(path, header=None).to_numpy()
def read_temperature(path):
    return pd.read_csv(path)['Temperature (K)'].values
def read_gridfile(path):
    return pd.read_csv(path)


class DmdJobInitalizer:
    def __init__(self, job, parent_project_workspace:str, bc_mapper=None):
        self.job = job
        self.parent_job = get_parent_job(parent_project_workspace, job, job.sp.cfd_sp)
        self.bc_mapper = bc_mapper

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
    def X_matrix(self) -> np.ndarray:
        path = Path(self.job.workspace()) / 'dmd'/'X_matrix.csv'
        array = read_csv(path)
        return array

    @property
    def coord(self) -> pd.DataFrame:
        path = Path(self.parent_job.workspace()) /"TimeSeries" / 'coord.csv'
        assert path.exists(), FileExistsError
        return read_gridfile(path)

    def T_wall(self, theta_deg: float) -> np.ndarray:
        theta_rad = theta_deg / 180 * np.pi
        loc = find_loc_index(theta_rad, self.coord)
        path = Path(self.parent_job.workspace()) / "TimeSeries" / f'loc_{loc}.csv'
        assert path.exists(), FileExistsError
        return read_temperature(path)