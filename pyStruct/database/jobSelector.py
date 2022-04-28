"""
This module select the cfd data based on the filtered condition
Two abstract class: 
    JobContainer
    DatabaseSelector

"""
import abc
from typing import Callable
from dataclasses import dataclass

ComposableJobs = Callable

@dataclass
class JobContainer(abc.ABC):
    @property
    @abc.abstractmethod
    def jobs(self):
        raise NotImplementedError


class DatabaseSelector(abc.ABC):
    @property
    @abc.abstractmethod
    def project(self):
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def jobs(self) -> JobContainer:
        raise NotImplementedError
    
    @abc.abstractmethod
    def filter(self, *fns: list[Callable]) -> JobContainer:
        """ Piping filtering functions """
        raise NotImplementedError
    
        