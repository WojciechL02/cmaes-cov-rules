import csv
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


class CMAESLogger:
    def __init__(self, results_path, func, dim, output_prefix="cmaes"):
        self.output_prefix = output_prefix  # experiment prefix (for files)
        self.results_path = results_path  # logs main dir path
        self.save_dir = None  # dir path for storing exp files
        self.log_data = []
        self.start_time = None
        self.end_time = None
        self.func = func
        self.dim = dim

    def start_logging(self):
        self.start_time = datetime.now()
        exp_dirname = f"{self.output_prefix}_{self.start_time}"
        self.save_dir = os.path.join(self.results_path, self.func, f"d{self.dim}", exp_dirname)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print(f"Optimization started at: {self.start_time}")

    def log(self, evals, fitness_values, optimizer):
        fbest = min(fitness_values)
        fmedian = np.median(fitness_values)
        fworst = max(fitness_values)
        sigma = optimizer._sigma
        self.log_data.append([evals, fbest, fmedian, fworst, sigma])

    def end_logging(self, seed):
        self.end_time = datetime.now()
        print(f"Optimization ended at: {self.end_time}")
        print(f"Total duration: {self.end_time - self.start_time}")
        self.save_log(seed)
        self.plot_results()

    def save_log(self, seed):
        filename = f"{self.output_prefix}_log_{seed}.csv"
        with open(os.path.join(self.save_dir, filename), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["evals", "fbest", "fmedian", "fworst", "sigma"])
            writer.writerows(self.log_data)

    def load_log(self):
        filename = f"{self.output_prefix}_log.csv"
        with open(os.path.join(self.save_dir, filename), "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            self.log_data = [list(map(float, row)) for row in reader]

    def plot_results(self, function_optimum: float = None):
        if not self.log_data:
            self.load_log()
        log_data = np.array(self.log_data)
        evals, fbest, fmedian, fworst, sigma = log_data.T

        plt.figure(figsize=(8, 5))
        plt.plot(evals, fbest, label="Best f(x)", color="blue")
        if function_optimum is not None:
            plt.plot(evals, function_optimum * np.ones_like(evals), label="Global optimum", color="black",
                     linestyle='dashed', linewidth=1)
        plt.yscale("log")
        plt.xlabel("Evaluations")
        plt.ylabel("Function Value")
        plt.title("Convergence Curve")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, f"{self.output_prefix}_convergence.png"))

        plt.figure(figsize=(8, 5))
        plt.plot(evals, fbest, label="Best f(x)", color="blue")
        plt.plot(evals, fmedian, label="Median f(x)", color="green")
        plt.plot(evals, fworst, label="Worst f(x)", color="red")
        if function_optimum is not None:
            plt.plot(evals, function_optimum * np.ones_like(evals), label="Global optimum", color="black", linestyle='dashed', linewidth=1)
        plt.yscale("log")
        plt.xlabel("Evaluations")
        plt.ylabel("Function Value")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, f"{self.output_prefix}_stats.png"))


def generate_distinct_colors(n):
    cmap = plt.get_cmap("hsv")
    return [cmap(i / n) for i in range(n)]


def plot_ecdf(log_files, func, dim, average=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    if average:
        # --- Aggregate all final fbest values at the last step of each run
        all_final_vals = []
        for log_file in log_files:
            with open(log_file, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                log_data = [list(map(float, row)) for row in reader]
            log_data = np.array(log_data)
            final_val = log_data[-1, 1]  # Last row, fbest
            all_final_vals.append(final_val)

        # --- Compute ECDF over the final best values
        sorted_vals = np.sort(all_final_vals)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        ax.plot(sorted_vals, ecdf, linestyle="-", linewidth=2, label="Average ECDF", color="black", marker="o", markersize=3)

    else:
        colors = generate_distinct_colors(len(log_files))

        for i, log_file in enumerate(log_files):
            seed = re.match(r".*_(.*).csv", log_file).group(1)
            with open(log_file, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                log_data = [list(map(float, row)) for row in reader]

            log_data = np.array(log_data)
            final_values = log_data[:, 1]  # Best function values at each evaluation step
            sorted_vals = np.sort(final_values)
            ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

            label = os.path.dirname(log_file).split("/")[3].split(".")[0]
            label = f"cmaes_mod_{seed}" if label.startswith("cmaes_mod") else f"cmaes_{seed}"
            ax.plot(sorted_vals, ecdf, marker="o", linestyle="-", label=label,
                    linewidth=0.5, markersize=1, color=colors[i])

        # --- Legend outside only if not averaging
        num_cols = min(2, max(2, len(log_files) // 10))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5),
                  fontsize="x-small", ncol=num_cols, title="Seeds")

    # --- Common settings
    ax.set_xscale("log")
    ax.set_xlabel("Function Value (log scale)")
    ax.set_ylabel("ECDF")
    ax.set_title(f"ECDF (CEC2017-{func} d={dim})")
    ax.grid()

    os.makedirs("ecdfs", exist_ok=True)
    suffix = "avg" if average else "all"
    plt.savefig(os.path.join("ecdfs", f"ecdf_{func}_{dim}_{suffix}.png"), bbox_inches="tight")
    return fig, ax