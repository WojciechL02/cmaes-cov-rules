import argparse

import numpy as np
import opfunu
import matplotlib.pyplot as plt

from cmaes import CMA
from cmaes_mod import CMA_Mod
from logger import CMAESLogger


def plot_debug_info(data: dict, name: str):
    fig, (ax1, ax2) = plt.subplots(2)
    N = len(data["condition"])
    ax1.plot(list(range(N)), data["condition"])
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Condition number")
    ax1.set_title("C matrix condition")
    ax1.grid()

    ax2.plot(list(range(N)), data["log_min_eigval"], label="Min", color="red")
    ax2.plot(list(range(N)), data["log_median_eigval"], label="Median", color="green")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Log eigval")
    ax2.set_title("Log10 eigenvalues stats")
    ax2.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name}.pdf")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CMA-ES optimization.")
    parser.add_argument("--cmaes-type", type=str, default="base", help="CMA-ES algorithm type..")
    parser.add_argument("--dim", type=int, default=10, help="Objective function dimensionality.")
    parser.add_argument("--function", type=int, default=1, help="CEC2017 function number.")
    parser.add_argument("--debug", action="store_true", help="Debug C matrix.")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = 42
    np.random.seed(seed)

    dim = args.dim

    for func_class in opfunu.get_functions_based_classname("2017"):
        f = func_class(ndim=dim)
        if f"F{args.function}:" in f.name:
            print("-" * 20)
            print("Optimizing:", f.name)
            f_name = f.name.split(":")[0]

            mean = np.random.uniform(-100, 100, dim)
            sigma = 1.0
            if args.cmaes_type == "base":
                optimizer = CMA(seed=seed, mean=mean, sigma=sigma, debug=args.debug)
            elif args.cmaes_type == "mod":
                optimizer = CMA_Mod(seed=seed, mean=mean, sigma=sigma, history=1000, debug=args.debug)
            else:
                raise ValueError(f"Unknown cmaes type: {args.cmaes_type}")

            output = "cmaes" if args.cmaes_type == "base" else "cmaes_mod"
            logger = CMAESLogger("debug", func=f_name, dim=dim, output_prefix=output)
            logger.start_logging(args.cmaes_type)

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

                # if evals % 3000 == 0:
                #     print(f"{evals:5d}  {min(fitness_values):10.5f}")

                if optimizer.should_stop():
                    break

            logger.end_logging(seed, args.cmaes_type)
            logger.plot_results(function_optimum=f.f_global)
            plot_debug_info(optimizer.debug_data, args.cmaes_type)


if __name__ == "__main__":
    main()
