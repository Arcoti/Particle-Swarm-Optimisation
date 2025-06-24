import numpy as np

from .base import BaseTestFunction

class Ackley(BaseTestFunction):
    """
    The Ackley Function.
    """
    def __init__(self, n: int, a: int = 20, b: float = 0.2, c: float = np.pi * 2, domain: tuple[float, float] = (-32.768, 32.768)):
        self._a = a
        self._b = b
        self._c = c
        self._n = n
        self._domain = domain

    def __call__(self, x: np.ndarray):
        if x.size != self._n:
            raise ValueError(f"Input dimension {x.size} does not match expected {self._n}")
        
        if not self.within_domain(x):
            raise ValueError("Input values does not fall within the expected domain")
        
        term_one = np.exp( - self._b * np.sqrt( np.sum(x**2) / x.size ) )
        term_two = np.exp( np.sum( np.cos(self._c * x) ) / x.size )
        
        return - self._a * term_one - term_two + self._a + np.exp(1)
    
    @property
    def global_minimum(self):
        return np.zeros(self._n), 0
    
    @property
    def domain(self):
        return self._domain
    
    def within_domain(self, x: np.ndarray):
        return np.all( (x >= self._domain[0]) & (x <= self._domain[1]))