import numpy as np
import opfunu
from tqdm import tqdm
import wandb
import argparse
import os
import json

from cmaes import CMA
from cmaes_mod import CMA_Mod
from wandb_logger import Logger


def main(seed: int, dim: int, method: str):
    np.random.seed(seed)

    exp_name = f"{method}_dim{dim}"
    max_evals = 10000 * dim

    config = {
        "seed": seed,
        "dim": dim,
        "method": method,
        "max_evals": max_evals,
    }
    if method == "mod":
        history = 2 * dim
        config["history"] = history

    logger = Logger(exp_name)
    logger.log_args(config)

    results = {}

    for func_class in tqdm(opfunu.get_functions_based_classname("2017"), desc=f"Seed {seed}, Dim {dim}"):
        f = func_class(ndim=dim)
        f_name = f.name.split(":")[0]

        mean = np.random.uniform(-100, 100, dim)
        sigma = 1.0
        if method == "base":
            optimizer = CMA(seed=seed, mean=mean, sigma=sigma)
        elif method == "mod":
            optimizer = CMA_Mod(seed=seed, mean=mean, sigma=sigma, history=history)
        else:
            raise ValueError(f"CMA-ES type {method} not known.")

        evals = 0
        per_generation_best_fitness = []

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
            fbest = min(fitness_values)
            fmedian = np.median(fitness_values)
            fworst = max(fitness_values)
            logger.log_scalar(evals, fbest, f_name, "convergence")

            # Log the best fitness this generation
            per_generation_best_fitness.append(fbest)

            if optimizer.should_stop():
                break
        
        results[f_name] = {
            "best": fbest,
            "median": fmedian,
            "worst": fworst,
        }

        # --- Compute ECDF from generation-wise bests
        fitness_array = np.array(per_generation_best_fitness)
        fitness_array[fitness_array <= 0] = 1e-12  # Avoid log(0)

        sorted_vals = np.sort(fitness_array)
        ecdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        log_xs = np.log10(sorted_vals)

        logger.log_ecdf(log_xs, ecdf_y, f_name)

        wandb.run.summary["final_best_fitness"] = float(fitness_array[-1])
        wandb.run.summary["global_optimum"] = f.f_global
    
    file_path = f"results/{method}_{dim}/{seed}.json"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dim", type=int, choices=[10, 30, 50, 100])
    parser.add_argument("--method", type=str, choices=["base", "mod"])
    args = parser.parse_args()

    main(args.seed, args.dim, args.method)
