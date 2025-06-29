import numpy as np
from tqdm import tqdm

from .evaluate import normalized_average_loss
from ..benchmark.base import BaseTestFunction

# Define Global Variables
total_particles = 50        # Total Number of Particles
total_iterations = 100      # Total Number of Iterations

w_max, w_min = 0.9, 0.2     # Inertia Weight
c1, c2 = 1.5, 1.5           # Standard Coefficent
k = 0.2                     # Scaling Factor

def evaluate_pso(dimension: int, function: BaseTestFunction, verbose=True, average_loss=False):
    np.set_printoptions(precision=2)

    # Initialization
    low, high = function.domain
    particles = np.random.uniform(low=low, high=high, size=(total_particles, dimension))

    # Define particle and global best
    particles_best = np.copy(particles)
    global_best = find_global_best(particles_best, function)

    # Define initial velocities
    max_velocity = k * (high - low)
    velocities = np.random.uniform(low=(- max_velocity), high=max_velocity, size=(total_particles, dimension))

    pbar = tqdm(range(total_iterations), desc="Progress", dynamic_ncols=True, unit="step") if verbose else range(total_iterations)
    for iteration in pbar:

        # Generate the r1 and r2
        r1 = np.random.uniform(size=(total_particles, dimension))
        r2 = np.random.uniform(size=(total_particles, dimension))

        # Get Inertia Weight 
        w = w_max - ((w_max - w_min) / total_iterations) * iteration

        # Update the particle's position
        velocities = np.clip((w * velocities + c1 * r1 * (particles_best - particles) + c2 * r2 * (global_best - particles)), -max_velocity, max_velocity)
        particles = particles + velocities

        # Update particle and global best
        particles_best = update_particle_best(particles, particles_best, function)
        global_best = find_global_best(particles_best, function)

        # Evaluate Average Loss
        if average_loss:
            normalized_average_loss(particles, function)
    
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
