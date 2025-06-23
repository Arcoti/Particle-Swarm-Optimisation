import numpy as np

def rastrigin(x):
    """
    The Rastrigin Function. Takes in n-dimensional input array, x, and then outputs the output.
    """
    # Define constant
    A = 10
    # Rastrigin function
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))