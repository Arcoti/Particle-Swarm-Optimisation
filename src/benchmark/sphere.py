import numpy as np

from .base import BaseTestFunction

class Sphere(BaseTestFunction):
    """
    The Sphere Function.
    """
    
    def __init__(self, n: int, domain: tuple[float, float] = (-5.12, 5.12)):
        self._n = n
        self._domain = domain

    def __call__(self, x: np.ndarray):
        if x.size != self._n:
            raise ValueError(f"Input dimension {x.size} does not match expected {self._n}")

        return np.sum( x** 2 )
    
    @property
    def global_minimum(self):
        return np.zeros(self._n), 0
    
    @property
    def domain(self):
        return self._domain
    
    def within_domain(self, x: np.ndarray):
        return np.all ( (x >= self._domain[0]) & (x <= self._domain[1]) )
    