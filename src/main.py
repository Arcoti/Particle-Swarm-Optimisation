from .benchmark.rastrigin import Rastrigin
from .PSO import particle_swarm_optimisation

if __name__ == "__main__":
    n = 2
    rastrigin = Rastrigin(n)
    result = particle_swarm_optimisation(n, rastrigin, verbose=True)
    print(result)