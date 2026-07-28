import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import interpolate
from torch.cuda.amp import autocast
from matplotlib.patches import Patch


import sys
sys.path.insert(0, '../../')
from simulator import TU_IM_Runner, TUIM_registry 
from AgentTorch.helpers import read_config

# *************************************************************************
# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="AgentTorch: design, simulate and optimize agent-based models"
)
parser.add_argument(
    "-c", "--config", help="Name of the yaml config file with the parameters."
)
# *************************************************************************

args = parser.parse_args()
config_file = args.config

print("Config file: ", config_file)

config = read_config(config_file)
registry = TUIM_registry()

runner = TU_IM_Runner(config, registry)
device = torch.device(runner.config['simulation_metadata']['device'])
runner.init()

num_steps_per_episode = runner.config["simulation_metadata"]["num_steps_per_episode"]

print('InProgress: TODO - execute the simulations!!')

def plot_grids(initial_grid, final_grid):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Define cooler colors
    colors = ['white', '#ffcc99', '#66b3ff']  # white for normal, light orange for tumor, light blue for immune
    cmap = plt.cm.colors.ListedColormap(colors)
    bounds = [0, 1, 2, 3]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot initial grid
    cax1 = ax1.imshow(initial_grid, cmap=cmap, norm=norm)
    ax1.set_title('Initial Grid', fontsize=14)
    
    # Plot final grid
    cax2 = ax2.imshow(final_grid, cmap=cmap, norm=norm)
    ax2.set_title('Final Grid', fontsize=14)
    
    # Add legend with cooler colors
    legend_elements = [
        Patch(facecolor='#ffcc99', label='Tumor Cells'),  # Light orange for Tumor Cells
        Patch(facecolor='#66b3ff', label='Immune Cells')  # Light blue for Immune Cells
    ]
    
    ax1.legend(handles=legend_elements, loc='upper right', fontsize='small', title='Cell Types')
    ax2.legend(handles=legend_elements, loc='upper right', fontsize='small', title='Cell Types')
    
    plt.tight_layout()
    plt.show()
    plt.close()

def calculate_RDF(grid):
    rows, cols = grid.shape
    center = torch.tensor([rows/2, cols/2], device=grid.device, dtype=torch.float32)
    max_distance = torch.sqrt(torch.tensor(rows**2 + cols**2, dtype=torch.float32, device=grid.device)) / 2
    r = torch.linspace(0, max_distance, 100, device=grid.device, dtype=torch.float32)
    
    cell_positions = torch.nonzero(grid != 0, as_tuple=False).float()
    
    rdf = torch.zeros_like(r)
    for i in range(len(r)):
        if i == 0:
            mask = torch.sum((cell_positions - center)**2, dim=1) <= r[i]**2
        else:
            mask = (torch.sum((cell_positions - center)**2, dim=1) > r[i-1]**2) & (torch.sum((cell_positions - center)**2, dim=1) <= r[i]**2)
        rdf[i] = torch.sum(mask)
    
    area = torch.pi * (r**2 - torch.cat((torch.tensor([0], device=grid.device, dtype=torch.float32), r[:-1]))**2)
    rdf = rdf / area
    rdf = rdf / torch.mean(rdf[-10:])
    
    return rdf, r

def torch_interp(x, xp, fp):
    x = x.to(torch.float32)
    xp = xp.to(torch.float32)
    fp = fp.to(torch.float32)

    sorted_indices = torch.argsort(xp)
    xp = xp[sorted_indices]
    fp = fp[sorted_indices]

    x = torch.clamp(x, xp[0], xp[-1])

    idxs_below = torch.searchsorted(xp, x, right=False) - 1
    idxs_above = torch.searchsorted(xp, x, right=True)

    idxs_below = torch.clamp(idxs_below, 0, len(xp) - 2)
    idxs_above = torch.clamp(idxs_above, 1, len(xp) - 1)

    x_below = xp[idxs_below]
    x_above = xp[idxs_above]
    y_below = fp[idxs_below]
    y_above = fp[idxs_above]

    slope = (y_above - y_below) / (x_above - x_below)
    return y_below + slope * (x - x_below)

def interpolate_RDFs(rdf1, rdf2, r1, r2):
    r_min = torch.max(torch.min(r1), torch.min(r2))
    r_max = torch.min(torch.max(r1), torch.max(r2))
    r_common = torch.linspace(r_min, r_max, 100, device=r1.device, dtype=torch.float32)
    
    rdf1_interp = torch_interp(r_common, r1, rdf1)
    rdf2_interp = torch_interp(r_common, r2, rdf2)
    
    return rdf1_interp, rdf2_interp, r_common

def calculate_cluster_size(L, cell_type):
    from scipy import ndimage
    labeled, num_features = ndimage.label(L == cell_type)
    if num_features == 0:
        return 0
    return np.mean(np.bincount(labeled.ravel())[1:])

def calculate_metrics(initial_grid, final_grid):
    rdf_initial, r_initial = calculate_RDF(initial_grid)
    rdf_final, r_final = calculate_RDF(final_grid)
    
    rdf_initial, rdf_final, r = interpolate_RDFs(rdf_initial, rdf_final, r_initial, r_final)
    
    tumor_fraction_initial = torch.sum(initial_grid == 1).float() / initial_grid.numel()
    tumor_fraction_final = torch.sum(final_grid == 1).float() / final_grid.numel()
    immune_fraction_initial = torch.sum(initial_grid == 2).float() / initial_grid.numel()
    immune_fraction_final = torch.sum(final_grid == 2).float() / final_grid.numel()
    
    metrics = {
        'rdf_initial': rdf_initial,
        'rdf_final': rdf_final,
        'r': r,
        'tumor_fraction_initial': tumor_fraction_initial,
        'tumor_fraction_final': tumor_fraction_final,
        'immune_fraction_initial': immune_fraction_initial,
        'immune_fraction_final': immune_fraction_final
    }
    
    return metrics

def plot_metrics(metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Define cooler colors
    initial_color = '#66b3ff'  # Light blue for Initial
    final_color = '#99ff99'    # Light green for Final
    
    # Plot RDF comparison
    ax1.plot(metrics['r'], metrics['rdf_initial'], color=initial_color, label='Initial')
    ax1.plot(metrics['r'], metrics['rdf_final'], color=final_color, label='Final')
    ax1.set_title(r'RDF Comparison: $RDF(r)$ vs. $r$', fontsize=14)
    ax1.set_xlabel(r'$r$', fontsize=12)
    ax1.set_ylabel(r'$RDF$', fontsize=12)
    
    # Plot cell fractions
    fractions = [
        [metrics['tumor_fraction_initial'], metrics['tumor_fraction_final']],
        [metrics['immune_fraction_initial'], metrics['immune_fraction_final']]
    ]
    x = np.arange(2)
    width = 0.35
    ax2.bar(x - width/2, [f[0] for f in fractions], width, color=initial_color, label='Initial')
    ax2.bar(x + width/2, [f[1] for f in fractions], width, color=final_color, label='Final')
    ax2.set_ylabel('Fraction')
    ax2.set_title('Cell Fractions')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Tumor', 'Immune'])
    
    # Create a legend
    legend_elements = [
        Patch(facecolor=initial_color, label='Initial'),
        Patch(facecolor=final_color, label='Final')
    ]
    
    ax1.legend(handles=legend_elements, loc='upper right')
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    plt.close()

def display_results(metrics):
    print(f"Tumor fraction: {metrics['tumor_fraction_initial']:.4f} -> {metrics['tumor_fraction_final']:.4f}")
    print(f"Immune fraction: {metrics['immune_fraction_initial']:.4f} -> {metrics['immune_fraction_final']:.4f}")

# Run the substeps for simulations
runner.execute()

print('Simulation completed!')

# Calculate metrics
initial_grid = runner.get_initial_state() 
final_grid = runner.get_final_state()
metrics = calculate_metrics(initial_grid, final_grid)
# plot_grids(initial_grid, final_grid)

# Plot and display results
# plot_metrics(metrics)
display_results(metrics)

# import ipdb; ipdb.set_trace()