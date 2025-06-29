from tqdm import tqdm

from .benchmark import *
from .PSO import particle_swarm_optimisation
from .animation.animate import create_animation
from .evaluation.eval_pso import evaluate_pso
from .evaluation.evaluate import plot_average_loss, match, plot_success

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

def success_main(n_start: int = 2, n_end: int = 10):
    to_plot = {}

    for n in tqdm(range(n_start, n_end + 1), desc="Dimension Progress", dynamic_ncols=True, unit="step"):
        functions = [Ackley(n), Griewank(n), Michalewicz(n), Rastrigin(n), Rosenbrock(n), Sphere(n)]

        for function in functions:
            matches = []

            for i in tqdm(range(500), desc="Trials Progress", leave=False, dynamic_ncols=True, unit="step"):
                result = evaluate_pso(n, function, verbose=False)
                matches.append(1 if match(function, result) else 0)

            total = sum(matches)

            if function.__class__.__name__ in to_plot:
                to_plot[function.__class__.__name__].append(total)
            else:
                to_plot[function.__class__.__name__] = [total]

    plot_success(to_plot, n_start, n_end)

def main(n: int):
    """
    Run particle swarm optimisation for all benchmark functions.
    """
    functions = [Ackley(n), Griewank(n), Michalewicz(n), Rastrigin(n), Rosenbrock(n), Sphere(n)]

    for function in functions:
        particle_swarm_optimisation(n, function)

if __name__ == "__main__":
    success_main()