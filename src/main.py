from .benchmark import *
from .PSO import particle_swarm_optimisation
from .animation.animate import create_animation

if __name__ == "__main__":
    n = 2
    result = particle_swarm_optimisation(n, Sphere(n), animate=True)
    create_animation(Sphere(n))
    print(result)