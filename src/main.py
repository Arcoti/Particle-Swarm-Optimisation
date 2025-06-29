from .benchmark import *
from .PSO import particle_swarm_optimisation
from .animation.animate import create_animation
from .evaluation.eval_pso import evaluate_pso
from .evaluation.evaluate import plot_average_loss

def animate_main(n: int = 2):
    """
    Create animation gif for all benchmark functions.
    """
    functions = [Ackley(n), Griewank(n), Michalewicz(n), Rastrigin(n), Rosenbrock(n), Sphere(n)]

    for function in functions:
        particle_swarm_optimisation(n, function, animate=True)
        create_animation(function, filePath = f'./media/{function.__class__.__name__}.gif')

def evaluate_main(n: int = 2):
    """
    Evaluate average loss over time.
    """
    functions = [Ackley(n), Griewank(n), Michalewicz(n), Rastrigin(n), Rosenbrock(n), Sphere(n)]

    for function in functions:
        evaluate_pso(n, function, average_loss=True)
        plot_average_loss(function)

def main(n: int):
    """
    Run particle swarm optimisation for all benchmark functions.
    """
    functions = [Ackley(n), Griewank(n), Michalewicz(n), Rastrigin(n), Rosenbrock(n), Sphere(n)]

    for function in functions:
        particle_swarm_optimisation(n, function)

if __name__ == "__main__":
    evaluate_main(10)