from .benchmark import *
from .PSO import particle_swarm_optimisation

if __name__ == "__main__":
    n = 10
    result = particle_swarm_optimisation(n, Rastrigin(n), animate=False)
    print(result)