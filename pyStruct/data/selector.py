"""
This module select the cfd data based on the filtered condition
"""
from typing import Protocol, Callable
from types import FunctionType
from itertools import compress
import pandas as pd
import signac
from signac.contrib.job import Job

class Database(Protocol):
    def __str__(self):
        raise NotImplementedError()
    
def get_all_keys(d):
    for key, value in d.items():
        yield  str(key) 
        if isinstance(value, dict):
            yield from get_all_keys(value) 

def recurse(data, level):
    # if type(data) is dict:
    try:
        for (key, value) in data.items():
            print("-" * level + str(key))
            recurse(value, level + 1)
    except:
        pass

class DatabaseSelector:
    """ An easy wrapper of signac job manager
        All the jobs are stored in lists
    """
    def __init__(self, database_path: str):
        self.project = signac.get_project(root=database_path)
        self.jobs = list(self.project.find_jobs()) 

    def print_sp(self):
        """ Print the sp keys"""
        # job = self.jobs.next()
        job = self.jobs[0]
        recurse(job.sp, 1)
        # for x in recurse(job.sp, 1):
        #     print(x)
    
    def filter_pipe(self, jobs: list[Job], *fns: list[Callable]) -> list[Job]:
        """ Piping filtering functions """
        for fn in fns:
            jobs = self.filter_jobs(jobs, fn)
            print(len(jobs))
        return jobs
        

    @staticmethod
    def filter_jobs(jobs: list, filter_fn: Callable) -> Job:
        """ Filter the job by the filter_fn, which is a function returns True/False
            This static method force check if every job in <jobs> satisfies the filter_fn
            and return a list of T/F
        Usage Example:
            no_stress = lambda job: job.sp.model.STRESS_ON == False
            filtered_jobs = filtered_jobs(jobs, no_stress)

        Return:
            filtered_job_list: list of jobs 
        """
        assert isinstance(filter_fn, FunctionType), "The filter function should be a function. "
        filtered_condi = [filter_fn(job) for job in jobs] # return a series of true/false
        filtered_job_list = list(compress(jobs, filtered_condi))
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

        
        
        



