import os
from collections import defaultdict
from logger import plot_ecdf
import matplotlib.pyplot as plt


log_files_grouped = defaultdict(list)

# Walk through the "logs/" directory and group paths
for subdir, dirs, files in os.walk("logs/"):
    for file in files:
        if os.path.splitext(file)[1] == ".csv":
            # Extract function and dimension (e.g., F6, d10)
            parts = subdir.split(os.sep)  # Split path based on directory separators
            func = parts[-3]  # Assuming the function is 3 directories before the file
            dim = parts[-2].replace('d', '')  # Assuming dimensionality is the second-to-last folder (remove 'd')

            log_files_grouped[(func, dim)].append(os.path.join(subdir, file))


for i, (k, v) in enumerate(log_files_grouped.items()):
    if len(v) == 2:
        fig, ax = plot_ecdf(v, k[0], int(k[1]))
