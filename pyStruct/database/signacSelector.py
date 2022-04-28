from dataclasses import dataclass
from typing import Union, Callable
from types import FunctionType, LambdaType
from itertools import compress
import signac
from signac.contrib.job import Job

from pyStruct.database.jobSelector import JobContainer, DatabaseSelector


class OpenSignacProjectFails(Exception):
    def __init__(self, message=None):
        super(OpenSignacProjectFails, self).__init__(message)

@dataclass
class SignacJob:
    workspace: str
    parent_workspace: str
    sp_to_match: dict


@dataclass
class SignacJobContainer(JobContainer):
    _jobs: list[Job]

    @property
    def jobs(self):
        return self._jobs

    def __repr__(self):
        return '\n'.join([f'{job}: {job.sp}' for job in self._jobs]) + f'\ntotal jobs:{len(self._jobs)}'
    
    def __getitem__(self, i:int):
        return self._jobs[i]

    def __len__(self):
        return len(self._jobs)


def get_signac_parent_job(paraent_workspace: str, job, sp_to_match) -> Job:
    parent_project = signac.get_project(paraent_workspace)
    parent = list(parent_project.find_jobs(sp_to_match))
    if len(parent) == 0:
        raise ValueError(f"Job {job}: No parent is found.")
    elif len(parent) > 1:
        raise ValueError(f"Job {job}: More than one parent is found")
    return parent[0]


def recurse(data, level):
    # if type(data) is dict:
    try:
        for (key, value) in data.items():
            print("-" * level + str(key))
            recurse(value, level + 1)
    except:
        pass
        


class SignacDatabaseSelector(DatabaseSelector):
    """ An easy wrapper of signac job manager
        All the jobs are stored in lists
    """
    def __init__(self, workspace: str):
        try:
            self._project = signac.get_project(root=workspace)
        except:
            raise OpenSignacProjectFails
        
        self._jobs = SignacJobContainer(
            list(self._project.find_jobs()),
            )
    
    @property
    def project(self):
        return self._project
    
    @property
    def jobs(self):
        return self._jobs

    def __repr__(self):
        """ Print all jobs"""
        return '\n'.join([f'{job}: {job.sp}' for job in self._jobs]) + f'\n total jobs:{len(self._jobs)}'
    
    def filter(self, fns: list[Callable]) -> JobContainer:
        """ Piping filtering functions """
        assert type(fns) == list, "Group the functions in a list"
        for fn in fns:
            jobs = self._filter_jobs(self._jobs, fn)
        return jobs

    @staticmethod
    def _filter_jobs(jobs: SignacJobContainer, filter_fn: Callable) -> JobContainer:
        """ Filter the job by the filter_fn, which is a function returns True/False
            This static method force check if every job in <jobs> satisfies the filter_fn
            and return a list of T/F
        Usage Example:
            no_stress = lambda job: job.sp.model.STRESS_ON == False
            filtered_jobs = filtered_jobs(jobs, no_stress)

        Return:
            filtered_job_list: list of jobs 
        """
        assert isinstance(filter_fn, FunctionType), f"The filter function should be a function. "
        filtered_condi = [filter_fn(job) for job in jobs._jobs] # return a series of true/false

        # make sure something
        if not any(filtered_condi):
            raise ValueError("The filter is not working")

        filtered_job_list = SignacJobContainer(list(compress(jobs, filtered_condi)))
        return filtered_job_list
    
    @staticmethod
    def print_jobs(jobs: list):
        """ Print the jobs in a human readable way"""
        for job in jobs:
            print(f'{job}: {job.sp}')
    
    @staticmethod
    def write_jobs(filepath: str, jobs: list):
        """ Save the jobs in a human readable way """
        with open(filepath, 'w') as f:
            for job in jobs:
                f.write(f'{job}: {job.sp} \n')
        
        
        


