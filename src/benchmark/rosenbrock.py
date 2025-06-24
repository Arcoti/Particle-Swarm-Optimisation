import numpy as np

from .base import BaseTestFunction

class Rosenbrock(BaseTestFunction):
    """
    The Rosenbrock Function.
    """

    def __init__(self, n: int, a: int = 100, domain: tuple[float, float] = (-5, 10)):
        self._a = a
        self._n = n
        self._domain = domain

    def __call__(self, x: np.ndarray):
        if x.size != self._n:
            raise ValueError(f"Input dimension {x.size} does not match expected {self._n}")
        
        if not self.within_domain(x):
            raise ValueError("Input values does not fall within the expected domain")
        
        return np.sum(self._a * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    @property
    def global_minimum(self):
        return np.ones(self._n), 0
    
    @property
    def domain(self):
        return self._domain

    def within_domain(self, x):
        return np.all( (x >= self._domain[0]) & (x <= self._domain[1]) )
    