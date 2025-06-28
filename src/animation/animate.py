import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.animation import FuncAnimation

from ..benchmark.base import BaseTestFunction

# Global Array to store particle values.
particles_against_time = []

fig, ax = plt.subplots()

def contour_plot(function: BaseTestFunction, nx = 100, ny = 100):
    low, high = function.domain
    x = np.linspace(low, high, nx)
    y = np.linspace(low, high, ny)
    X, Y = np.meshgrid(x, y)

    points_grid = np.vstack([X.ravel(), Y.ravel()]).T
    Z = np.apply_along_axis(function, axis=1, arr=points_grid)
    Z = Z.reshape(ny, nx)

    return ax.contour(X, Y, Z, levels=30, cmap='viridis')

def initial_plot(function: BaseTestFunction, particles: np.ndarray):
    contour = contour_plot(function)
    scatter = ax.scatter(particles[:, 0], particles[:, 1], color='red', zorder=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Particle Swarm Optimisation")

    plt.ion()
    plt.show()

    return scatter

def update_points(particles: np.ndarray, scatter: PathCollection):
    particles_against_time.append(particles)
    scatter.set_offsets(particles)
    fig.canvas.draw()
    fig.canvas.flush_events()

def interactive_off():
    plt.ioff()
    plt.show()

# Creation of Animation
# ---------------------

def animation_update(frame: int, scatter: PathCollection):
    scatter.set_offsets(particles_against_time[frame + 1])
    return scatter, # Returns as a tuple

def create_animation(function: BaseTestFunction):
    if particles_against_time == []:
        raise ValueError("No data collected for animation to run")
    
    scatter = initial_plot(function, particles_against_time[0])
    ani = FuncAnimation(fig, animation_update, frames=99, fargs=(scatter, ), interval=50, blit=True)
    ani.save('./media/animation.gif', fps=30)