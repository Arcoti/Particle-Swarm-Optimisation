import numpy as np

from .base import BaseTestFunction

class Michalewicz(BaseTestFunction):
    """
    The Michalewicz Function.
    """
    def __init__(self, n: int, m: int = 10, domain: tuple[float, float] = (0, np.pi)):
        self._m = m
        self._n = n
        self._domain = domain

    def __call__(self, x: np.ndarray):
        if x.size != self._n:
            raise ValueError(f"Input dimension {x.size} does not match expected {self._n}")
        
        i = np.arange(1, x.size + 1)
        return - np.sum( np.sin(x) * ( np.sin((i * x**2) / np.pi) )**self._m )
    
    @property
    def global_minimum(self):
        return np.full(self._n, np.pi), -self._n
    
    @property
    def domain(self):
        return self._domain
    
    def within_domain(self, x):
        return np.all( (x >= self._domain[0]) & (x <= self._domain[1] ) )
    