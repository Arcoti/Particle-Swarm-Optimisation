from .benchmark import *
from .PSO import particle_swarm_optimisation
from .animation.animate import create_animation

def animate_main(n: int = 2):
    """
    Create animation gif for all benchmark functions.
    """
    functions = [Ackley(n), Griewank(n), Michalewicz(n), Rastrigin(n), Rosenbrock(n), Sphere(n)]

    for function in functions:
        particle_swarm_optimisation(n, function, animate=True)
        create_animation(function, filePath = f'./media/{function.__class__.__name__}.gif')

def main(n: int):
    """
    Run particle swarm optimisation for all benchmark functions.
    """
    functions = [Ackley(n), Griewank(n), Michalewicz(n), Rastrigin(n), Rosenbrock(n), Sphere(n)]

    for function in functions:
        particle_swarm_optimisation(n, function)

if __name__ == "__main__":
    animate_main()