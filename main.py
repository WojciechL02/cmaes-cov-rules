import argparse
import numpy as np
from cmaes_mod import CMA
from logger import CMAESLogger
from opfunu.cec_based.cec2017 import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run CMA-ES optimization.")
    parser.add_argument("--dim", type=int, default=10, help="Objective function dimensionality.")
    parser.add_argument("--mean", type=float, default=3.0, help="Initial mean for CMA-ES.")
    parser.add_argument("--sigma", type=float, default=2.0, help="Initial step size for CMA-ES.")
    parser.add_argument("--results-path", type=str, default="logs/")
    parser.add_argument("--output", type=str, default="cmaes", help="Prefix for output files.")
    return parser.parse_args()


def main():
    args = parse_args()

    seed = 42
    np.random.seed(seed)

    dim = args.dim

    f = F12017(ndim=dim)

    optimizer = CMA(seed=seed, mean=3 * np.ones(dim), sigma=2.0, alpha_hist=0.5)
    # logger = CMAESLogger(args.results_path)
    # logger.start_logging()

    evals = 0
    while True:
        solutions = []
        fitness_values = []

        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = f.evaluate(x)
            evals += 1
            solutions.append((x, value))
            fitness_values.append(value)

        optimizer.tell(solutions)
        # logger.log(evals, fitness_values, optimizer)

        if evals % 3000 == 0:
            print(f"{evals:5d}  {min(fitness_values):10.5f}")

        if optimizer.should_stop():
            break

    # logger.end_logging()
    # logger.plot_results()


if __name__ == "__main__":
    main()
