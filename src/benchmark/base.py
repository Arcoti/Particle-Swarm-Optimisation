from numpy import ndarray, bool
from abc import ABC, abstractmethod

class BaseTestFunction(ABC):
    """
    An interface for test functions.
    """
    @abstractmethod
    def __call__(self, x: ndarray):
        ...

    @property
    @abstractmethod
    def domain(self) -> tuple[float, float]:
        ...

    @abstractmethod
    def within_domain(self, x: ndarray) -> bool:
        ...
