import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def get_data(method: str, dim: int):
    json_list = []
    for seed in range(51):
        path = f"results/{method}_{dim}/{seed}.json"
        with open(path, "r") as f:
            data = json.load(f)
            best_values = {func: values['best'] for func, values in data.items()}
            json_list.append(best_values)
    return json_list

dim = 30
data_base = get_data("base", dim)
data_mod = get_data("mod", dim)

df_base = pd.DataFrame(data_base)
df_mod = pd.DataFrame(data_mod)
df_base.sort_index(axis=1)
df_mod.sort_index(axis=1)

# pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
print("VANILLA:")
print(df_base.describe().loc[['mean', 'std']].to_string())
print("MODIFIED:")
print(df_mod.describe().loc[['mean', 'std']].to_string())

# Create a figure with a 6x5 grid of subplots (30 total spaces for 29 functions)
fig, axes = plt.subplots(6, 5, figsize=(20, 24))
axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

# Get all column names
all_cols = df_base.columns.tolist()

# Loop through each function and create a boxplot in its corresponding subplot
for i, col in enumerate(all_cols):
    if i < len(axes):  # Make sure we don't exceed the number of subplots
        ax = axes[i]
        
        # Prepare data for this function
        data = [df_base[col], df_mod[col]]
        
        # Create boxplot
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        
        # Color the boxes
        bp['boxes'][0].set(facecolor='lightblue')  # Base model
        bp['boxes'][1].set(facecolor='lightgreen')  # Modified model
        
        # Set title and customize subplot
        ax.set_title(f'{col}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Set x-tick labels
        ax.set_xticklabels(['Base', 'Mod'], fontsize=10)
        
        # Only add ylabel to the leftmost plots
        if i % 5 == 0:
            ax.set_ylabel('Value', fontsize=10)

# If we have less than 30 functions, turn off the unused subplots
for j in range(len(all_cols), len(axes)):
    axes[j].set_visible(False)

# Add a main title
plt.suptitle('Comparison of Base vs Modified CMA-ES on CEC2017', fontsize=16, y=0.995)

# Add a common legend for the entire figure
legend_elements = [
    Patch(facecolor='lightblue', edgecolor='black', label='Base'),
    Patch(facecolor='lightgreen', edgecolor='black', label='Modified')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
           ncol=2, fontsize=12)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Make room for the suptitle
plt.savefig(f"functions_grid_d{dim}.png", format="png", dpi=300)
plt.close()

