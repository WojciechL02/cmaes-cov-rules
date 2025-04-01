import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class CMAESLogger:
    def __init__(self, results_path, output_prefix="cmaes"):
        self.output_prefix = output_prefix  # experiment prefix (for files)
        self.results_path = results_path  # logs main dir path
        self.save_dir = None  # dir path for storing exp files
        self.log_data = []
        self.start_time = None
        self.end_time = None

    def start_logging(self):
        self.start_time = datetime.now()
        exp_dirname = f"{self.output_prefix}_{self.start_time}"
        self.save_dir = os.path.join(self.results_path, exp_dirname)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print(f"Optimization started at: {self.start_time}")

    def log(self, evals, fitness_values, optimizer):
        fbest = min(fitness_values)
        fmedian = np.median(fitness_values)
        fworst = max(fitness_values)
        sigma = optimizer._sigma
        self.log_data.append([evals, fbest, fmedian, fworst, sigma])

    def end_logging(self):
        self.end_time = datetime.now()
        print(f"Optimization ended at: {self.end_time}")
        print(f"Total duration: {self.end_time - self.start_time}")
        self.save_log()
        self.plot_results()

    def save_log(self):
        filename = f"{self.output_prefix}_log.csv"
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

    def plot_results(self):
        if not self.log_data:
            self.load_log()
        log_data = np.array(self.log_data)
        evals, fbest, fmedian, fworst, sigma = log_data.T

        plt.figure(figsize=(8, 5))
        plt.plot(evals, fbest, label="Best f(x)", color="blue")
        plt.xlabel("Evaluations")
        plt.ylabel("Best Function Value")
        plt.title("Convergence Curve")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, f"{self.output_prefix}_convergence.png"))

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(evals, fbest, label="Best f(x)", color="blue")
        plt.plot(evals, fmedian, label="Median f(x)", color="green")
        plt.plot(evals, fworst, label="Worst f(x)", color="red")
        plt.yscale("log")
        plt.xlabel("Evaluations")
        plt.ylabel("Function Value")
        plt.legend()
        plt.grid()

        # plt.subplot(2, 1, 2)
        # plt.plot(evals, sigma, label="Min std", color="cyan")
        # plt.yscale("log")
        # plt.xlabel("Evaluations")
        # plt.ylabel("Std / Axis Ratio")
        # plt.legend()
        # plt.grid()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{self.output_prefix}_stats.png"))


def plot_ecdf(log_files, save_dir="."):
    plt.figure(figsize=(8, 5))

    for log_file in log_files:
        with open(log_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            log_data = [list(map(float, row)) for row in reader]

        log_data = np.array(log_data)
        final_values = log_data[:, 1]  # Best function values at each evaluation step
        sorted_vals = np.sort(final_values)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        label = os.path.dirname(log_file).split("/")[1].split(".")[0]
        plt.plot(sorted_vals, ecdf, marker="o", linestyle="-", label=label, linewidth=0.5, markersize=1)
        plt.xscale("log")

    plt.xlabel("Num. fevals")
    plt.title("ECDF Comparison")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"ecdf_comparison.png"))
