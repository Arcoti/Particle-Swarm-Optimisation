import numpy as np
import matplotlib.pyplot as plt

from ..benchmark.base import BaseTestFunction

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

def plot_average_loss(function: BaseTestFunction, iterations: int = 100):
    global average_loss_over_time
    x = range(1, iterations + 1)
    y = average_loss_over_time

    plt.plot(x, y)
    
    plt.xlabel("Normalized Average Loss")
    plt.ylabel("Iterations")
    plt.title(f"{function.__class__.__name__} - Average Loss over Iterations")

    plt.grid(True)
    plt.show()

    plt.savefig(f'./media/graphs/{function.__class__.__name__}.png')

    average_loss_over_time = []