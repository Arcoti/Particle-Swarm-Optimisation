import numpy as np

from .base import BaseTestFunction

class Rastrigin(BaseTestFunction):
    """
    The Rastrigin Function.
    """
    def __init__(self, n: int, a: int = 10, domain: tuple[float, float] = (-5.12, 5.12)):
        self._a = a
        self._n = n
        self._domain = domain

    def __call__(self, x: np.ndarray):
        if x.size != self._n:
            raise ValueError(f"Input dimension {x.size} does not match expected {self._n}")
        
        if not self.within_domain(x):
            raise ValueError("Input values does not fall within the expected domain")

        return self._a * len(x) + np.sum(x**2 - self._a * np.cos(2 * np.pi * x))

    @property
    def global_minimum(self):
        return np.zeros(self._n), 0
    
    @property
    def domain(self):
        return self._domain
    
    def within_domain(self, x: np.ndarray):
        return np.all( (x >= self._domain[0]) & (x <= self._domain[1]) )
    