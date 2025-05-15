import numpy as np
import opfunu
from tqdm import tqdm
import wandb
import argparse

from cmaes_mod import CMA_Mod


def main(seed: int, dim: int):
    np.random.seed(seed)

    for func_class in tqdm(opfunu.get_functions_based_classname("2017"), desc=f"Seed {seed}, Dim {dim}"):
        f = func_class(ndim=dim)
        f_name = f.name.split(":")[0]
        max_evals = 10000 * dim

        wandb.init(
            project="cmaes-cov-rules",
            name=f"{f_name}-dim{dim}-seed{seed}",
            config={
                "seed": seed,
                "dim": dim,
                "function": f_name,
                "max_evals": max_evals
            },
            reinit=True
        )

        mean = np.random.uniform(-100, 100, dim)
        sigma = 1.0
        optimizer = CMA_Mod(seed=seed, mean=mean, sigma=sigma, history=30)

        evals = 0
        per_generation_min_fitness = []

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

            # Log the minimum fitness this generation
            per_generation_min_fitness.append(min(fitness_values))

            if optimizer.should_stop():
                break

        # --- Compute ECDF from generation-wise bests
        fitness_array = np.array(per_generation_min_fitness)
        fitness_array[fitness_array <= 0] = 1e-12  # Avoid log(0)

        sorted_vals = np.sort(fitness_array)
        ecdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        log_xs = np.log10(sorted_vals)

        wandb.log({
            "ECDF": wandb.plot.line_series(
                xs=log_xs.tolist(),
                ys=[ecdf_y.tolist()],
                keys=["ECDF"],
                title=f"ECDF (per-generation min fitness) - {f_name} d={dim} seed={seed}",
                xname="log10(Fitness)",
            )
        })

        # Summary: best value seen during generations
        wandb.run.summary["final_best_fitness"] = float(sorted_vals[0])
        wandb.run.summary["final_best_ecdf_value"] = float(ecdf_y[0])
        wandb.run.summary["function_optimum"] = f.f_global
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    dims = [10, 30, 50, 100]  # all ndims for F102017 problem

    for dim in dims:
        main(args.seed, dim)
