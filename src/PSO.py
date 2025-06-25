import time
import numpy as np
from tqdm import tqdm
from matplotlib.collections import PathCollection

from .benchmark.base import BaseTestFunction
from .animation.animate import initial_plot, update_points

# Define Global Variables
total_particles = 50        # Total Number of Particles
total_iterations = 100      # Total Number of Iterations

w = 0.659                   # Inertia Weight
c1, c2 = 2, 2               # Standard Coefficent
k = 0.2                     # Scaling Factor

def particle_swarm_optimisation(dimension: int, function: BaseTestFunction, verbose=False):
    np.set_printoptions(precision=2)
    scatter = None

    # Initialization
    low, high = function.domain
    particles = np.random.uniform(low=low, high=high, size=(total_particles, dimension))

    # Define particle and global best
    particles_best = np.copy(particles)
    global_best = find_global_best(particles_best, function)

    # Define initial velocities
    max_velocity = k * (high - low)
    velocities = np.random.uniform(low=(- max_velocity), high=max_velocity, size=(total_particles, dimension))

    # Initial Plot
    if verbose:
        scatter = initial_plot(function, particles)
        

    for iteration in tqdm(range(total_iterations), desc="Progress", dynamic_ncols=True, unit="step"):

        # Produce Contour Plot
        if verbose and type(scatter) == PathCollection:
            update_points(particles, scatter)
            time.sleep(0.2)

        # Generate the r1 and r2
        r1 = np.random.uniform(size=(total_particles, dimension))
        r2 = np.random.uniform(size=(total_particles, dimension))

        # Update the particle's position
        velocities = w * velocities + c1 * r1 * (particles_best - particles) + c2 * r2 * (global_best - particles)
        particles = particles + velocities

        # Update particle and global best
        particles_best = update_particle_best(particles, particles_best, function)
        global_best = find_global_best(particles_best, function)
    
    return global_best

def find_global_best(particles: np.ndarray, function: BaseTestFunction):
    values = np.apply_along_axis(function, axis=1, arr=particles)
    min_index = np.argmin(values)
    return particles[min_index]

def update_particle_best(particles: np.ndarray, particles_best: np.ndarray, function: BaseTestFunction):
    current_values = np.apply_along_axis(function, axis=1, arr=particles)
    best_values = np.apply_along_axis(function, axis=1, arr=particles_best)
    mask = best_values <= current_values
    return np.where(mask[:, None], particles_best, particles)
