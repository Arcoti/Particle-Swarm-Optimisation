from .benchmark import *
from .PSO import particle_swarm_optimisation

if __name__ == "__main__":
    n = 2
    result = particle_swarm_optimisation(n, Michalewicz(n), verbose=True)
    print(result)