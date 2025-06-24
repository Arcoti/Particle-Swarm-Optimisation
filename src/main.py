from .benchmark.rastrigin import Rastrigin
from .PSO import particle_swarm_optimisation

n = 3

rastrigin = Rastrigin(n)

n = 3

solution = particle_swarm_optimisation(n, rastrigin)
print(solution)