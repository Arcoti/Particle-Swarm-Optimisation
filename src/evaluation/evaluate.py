import numpy as np
import matplotlib.pyplot as plt

from ..benchmark.base import BaseTestFunction

multiple_functions = {}
average_loss_over_time = []

def match(function: BaseTestFunction, global_best: np.ndarray, rtol=1e-05, atol=1e-08):
    position, value = function.global_minimum
    return np.allclose(position, global_best, rtol=rtol, atol=atol)

def average_loss(particles: np.ndarray, function: BaseTestFunction):
    return np.mean(np.apply_along_axis(function, axis=1, arr=particles))

def normalized_average_loss(particles: np.ndarray, function: BaseTestFunction):
    position, value = function.global_minimum
    nal = average_loss(particles, function) - value
    average_loss_over_time.append(nal)

def store(function: BaseTestFunction):
    global average_loss_over_time, multiple_functions

    multiple_functions[function.__class__.__name__] = average_loss_over_time
    average_loss_over_time = []

def plot_average_loss(iterations: int = 100):
    x = range(1, iterations + 1)

    for key, value in multiple_functions.items():
        print(key, value)
        plt.plot(x, value, label=key)
    
    plt.xlabel("Normalized Average Loss")
    plt.ylabel("Iterations")
    plt.title("Normalized Average Loss Over Iterations")
    plt.legend()

    plt.grid(True)
    plt.show()