from .benchmark.sphere import Sphere
from .PSO import particle_swarm_optimisation

if __name__ == "__main__":
    n = 2
    sphere = Sphere(n)
    result = particle_swarm_optimisation(n, sphere, verbose=True)
    print(result)