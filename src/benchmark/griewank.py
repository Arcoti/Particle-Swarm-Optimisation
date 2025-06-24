import numpy as np

from .base import BaseTestFunction

class Griewank(BaseTestFunction):
    def __init__(self, n: int, a: int = 1, b: int = 4000, domain: tuple[float, float] = (-600, 600)):
        self._a = a
        self._b = b
        self._n = n
        self._domain = domain

    def __call__(self, x: np.ndarray):
        if x.size != self._n:
            raise ValueError(f"Input dimension {x.size} does not match expected {self._n}")
        
        if not self.within_domain(x):
            raise ValueError("Input values does not fall within the expected domain")
        
        i = np.sqrt( np.arange(1, x.size + 1) )
        term_one = ( 1 / self._b ) * np.sum( x**2 )
        term_two = np.prod( np.cos( x / i ))
        
        return self._a + term_one - term_two
    
    @property
    def global_minimum(self):
        return np.zeros(self._n), 0
    
    @property
    def domain(self):
        return self._domain
    
    def within_domain(self, x):
        return np.all( (x >= self._domain[0]) & (x <= self._domain[1] ) )
    