import time
import numpy as np
from tqdm import tqdm
from matplotlib.collections import PathCollection

from .benchmark.base import BaseTestFunction
from .animation.animate import initial_plot, update_points, interactive_off

# Define Global Variables
total_particles = 50        # Total Number of Particles
total_iterations = 100      # Total Number of Iterations

w_max, w_min = 0.9, 0.2     # Inertia Weight
c1, c2 = 2, 2               # Standard Coefficent
k = 0.2                     # Scaling Factor

def particle_swarm_optimisation(dimension: int, function: BaseTestFunction, animate=False):
    """
    Particle Swarm Optimisation. 
    
    Parameters
    ----------
    dimension : int
        The dimension of the function
    
    function : BaseTestFunction
        The function to optimise

    animate : bool
        Whether to show the animation of the optimisation. Only works when dimension = 2. 
        Default is False. 
    """
    if validate_animate(dimension, animate):
        raise ValueError(f"Animation cannot occur when dimension not equals 2")

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
    if animate:
        scatter = initial_plot(function, particles)

    for iteration in tqdm(range(total_iterations), desc="Progress", dynamic_ncols=True, unit="step"):

        # Produce Contour Plot
        if animate and type(scatter) == PathCollection:
            update_points(particles, scatter)
            time.sleep(0.2) # Introduce a Gap to allow visualization of the particles state

        # Generate the r1 and r2
        r1 = np.random.uniform(size=(total_particles, dimension))
        r2 = np.random.uniform(size=(total_particles, dimension))

        # Get Inertia Weight 
        w = w_max - ((w_max - w_min) / total_iterations) * iteration

        # Update the particle's position
        velocities = w * velocities + c1 * r1 * (particles_best - particles) + c2 * r2 * (global_best - particles)
        particles = particles + velocities

        # Update particle and global best
        particles_best = update_particle_best(particles, particles_best, function)
        global_best = find_global_best(particles_best, function)
    
    # Turn of interactive mode
    if animate:
        interactive_off()
    
    return global_best

def find_global_best(particles: np.ndarray, function: BaseTestFunction):
    """
    From the array of all particles' best position, evaluate their values base on the function.
    Then, find the minimum of the value and retrieve its position, becoming global best position. 
    """
    values = np.apply_along_axis(function, axis=1, arr=particles)
    min_index = np.argmin(values)
    return particles[min_index]

def update_particle_best(particles: np.ndarray, particles_best: np.ndarray, function: BaseTestFunction):
    """
    From an array of current positions and particles' best positions, find their corresonpding values. 
    Depending on their values, find the minimum and update the position as particle's best. 
    """
    current_values = np.apply_along_axis(function, axis=1, arr=particles)
    best_values = np.apply_along_axis(function, axis=1, arr=particles_best)
    mask = best_values <= current_values
    return np.where(mask[:, None], particles_best, particles)

def validate_animate(dimension: int, animate: bool):
    """
    Returns True if dimension == 2 and animate == True. 
    Returns False otherwise. 
    """
    return dimension == 2 if animate else True