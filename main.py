import argparse
import numpy as np
from cmaes import CMA
from cmaes_mod import CMA as CMAmod
from logger import CMAESLogger
from opfunu.cec_based import cec2017
import opfunu


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

    for func_class in opfunu.get_functions_based_classname("2017"):
        f = func_class(ndim=dim)
        if "F1" not in f.name:
            continue
        print("-" * 20)
        print("Optimizing:", f.name)
        f_name = f.name.split(":")[0]

        if "Hybrid" in f.name:
            continue

        mean = np.random.uniform(-100, 100, dim)
        sigma = 1.0
        # optimizer = CMA(seed=seed, mean=mean, sigma=sigma)
        optimizer = CMAmod(seed=seed, mean=mean, sigma=sigma, history=1000)

        # TODO:
        # zapis z 51 seed'ow
        # ploty ECDF z zagregowanych 51 run'ow

        logger = CMAESLogger(args.results_path, func=f_name, dim=dim, output_prefix=args.output)
        logger.start_logging()

        max_evals = 10000 * dim
        evals = 0
        while evals < max_evals:
            solutions = []
            fitness_values = []

            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = f.evaluate(x)
                evals += 1
                solutions.append((x, value))
                fitness_values.append(value)

            optimizer.tell(solutions)
            logger.log(evals, fitness_values, optimizer)

            if evals % 3000 == 0:
                print(f"{evals:5d}  {min(fitness_values):10.5f}")

            if optimizer.should_stop():
                break

        logger.end_logging()
        logger.plot_results(function_optimum=f.f_global)


if __name__ == "__main__":
    main()
