from .benchmark.ackley import Ackley
from .PSO import particle_swarm_optimisation

if __name__ == "__main__":
    n = 2
    ackley = Ackley(n)
    result = particle_swarm_optimisation(n, ackley, verbose=True)
    print(result)