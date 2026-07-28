"""
visualize_trajectory.py
========================
Standalone script for capturing and visualizing tumor-immune ABM dynamics.

Captures the state trajectory at each substep using the existing
TU_IM_Runner and TUIM_registry, then produces:

  1. Substep-level animation (GIF / MP4) with labels & cell fractions
  2. Side-by-side comparison with post-treatment ground truth
  3. RDF evolution over the trajectory
  4. Cell fraction evolution over substeps
  5. Per-substep qualitative grid snapshots (PNG)

USAGE
-----
  python visualize_trajectory.py --config <path_to_config.yaml>

This script does NOT modify any existing files in the repository.
"""

import sys, os, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import imageio

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, PROJECT_ROOT)
from simulator import TU_IM_Runner, TUIM_registry
from AgentTorch.helpers import read_config

CMAP = ListedColormap(['white', '#ffcc99', '#66b3ff'])
NORM = BoundaryNorm([0, 1, 2, 3], CMAP.N)

SUBSTEP_DISPLAY = {
    'tumor_cell_proliferation': 'Tumor Proliferation',
    'tumor_cell_migration': 'Tumor Migration',
    'immune_cell_proliferation': 'Immune Proliferation',
    'immune_cell_migration': 'Immune Migration',
    'tumor_immune_cell_interaction': 'Combat',
    'tumor_cell_death': 'Tumor Death',
    'immune_cell_death': 'Immune Death',
    'immune_cell_influx': 'Immune Influx', 
}

def state_to_grid(state, device='cpu'):
    tl = state['agents']['tumorcells']['TU_location_matrix']
    il = state['agents']['immunecells']['IM_location_matrix']
    t2d = torch.sum(tl, dim=0)
    i2d = torch.sum(il, dim=0)
    g = torch.zeros_like(t2d, device=device, dtype=torch.int)
    g[t2d > 0] = 1
    g[i2d > 0] = 2
    return g

def extract_trajectory_grids(runner):
    snames = [runner.config['substeps'][sk]['name'] for sk in runner.config['substeps'].keys()]
    # TU_IM_Runner stores lightweight 2D grids in grid_trajectory (not state_trajectory)
    traj = runner.grid_trajectory
    grids, labels = [], []
    for ep_states in traj:
        for sidx, grid in enumerate(ep_states):
            grids.append(grid.detach().cpu().numpy() if hasattr(grid, 'detach') else grid)
            if sidx == 0:
                labels.append('Initial (pre-treatment)')
            else:
                d = SUBSTEP_DISPLAY.get(snames[sidx - 1], snames[sidx - 1])
                labels.append('After: %s' % d)
    return grids, labels, snames


def grid_fractions(grid):
    t = np.sum(grid == 1) / grid.size
    i = np.sum(grid == 2) / grid.size
    return t, i


def calculate_rdf_np(grid):
    rows, cols = grid.shape
    cy, cx = rows / 2.0, cols / 2.0
    r = np.linspace(0, np.sqrt(rows**2 + cols**2) / 2.0, 100)
    pos = np.argwhere(grid != 0).astype(float)
    if len(pos) == 0:
        return np.zeros_like(r), r
    dists = np.sqrt(np.sum((pos - np.array([cy, cx]))**2, axis=1))
    rdf = np.zeros_like(r)
    for i in range(len(r)):
        if i == 0:
            mask = dists <= r[i]
        else:
            mask = (dists > r[i-1]) & (dists <= r[i])
        rdf[i] = np.sum(mask)
    area = np.pi * (r**2 - np.concatenate(([0], r[:-1]))**2)
    area = np.maximum(area, 1e-6)   # CHANGED: 1e-12 was too permissive — still let a single
                                     # cell blow the ratio up to ~1e6; use a larger floor,
                                     # or better: leave bin 0's rdf at 0 explicitly.
    rdf = rdf / area
    rdf[0] = 0.0   # CHANGED: r=0 bin is degenerate by construction, zero it explicitly
    rdf = rdf / (np.mean(rdf[-10:]) + 1e-12)
    return rdf, r


def load_ground_truth(config, config_path, dl='post_10percent'):
    """Load ground-truth post-treatment grid from CSVs in the same patient dir."""
    tum_rel = config['simulation_metadata'].get('Tum_dense', '')
    if not tum_rel:
        return None
    # Resolve tum_path to absolute using the config file location
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    tum_abs = os.path.abspath(os.path.join(cfg_dir, tum_rel))
    # Go up from pre_10percent dir to patient dir, then into post_10percent
    pre_dir = os.path.dirname(tum_abs)         # .../data/p10/pre_10percent
    patient_dir = os.path.dirname(pre_dir)     # .../data/p10
    post_dir = os.path.join(patient_dir, dl)   # .../data/p10/post_10percent
    tc = os.path.join(post_dir, 'Tum_dense_2.csv')
    ic = os.path.join(post_dir, 'CD8_dense.csv')
    if not os.path.isfile(tc) or not os.path.isfile(ic):
        print('      Ground truth CSVs not found at: %s' % post_dir)
        return None
    import pandas as pd
    N, M = config['simulation_metadata']['N'], config['simulation_metadata']['M']
    tum = pd.read_csv(tc, header=None).values.reshape(N, M)
    imm = pd.read_csv(ic, header=None).values.reshape(N, M)
    gt = np.zeros((N, M), dtype=int)
    gt[tum > 0] = 1
    gt[imm > 0] = 2
    print('      Loaded ground truth from: %s' % post_dir)
    return gt


def create_substep_animation(grids, labels, opath, fps=1.0, fmt='gif'):
    """Create an animation from grid states. fps -> duration(ms) conversion."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    images = []
    for idx in range(len(grids)):
        ax.clear()
        ax.imshow(grids[idx], cmap=CMAP, norm=NORM, interpolation='nearest')
        tf, imf = grid_fractions(grids[idx])
        ax.set_title('%s\nTumor: %.3f  |  Immune: %.3f' % (labels[idx], tf, imf),
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.legend(handles=[
            Patch(facecolor='#ffcc99', label='Tumor (%.2f)' % tf),
            Patch(facecolor='#66b3ff', label='Immune (%.2f)' % imf),
        ], loc='upper right', fontsize='small', framealpha=0.7)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(img)
    plt.close(fig)
    
    duration_ms = 1000.0 / fps if fps > 0 else 500
    
    if fmt == 'gif':
        imageio.mimsave(opath, images, format='GIF', duration=duration_ms, loop=0)
    else:
        try:
            imageio.mimsave(opath, images, format='FFMPEG', fps=fps)
        except Exception:
            gpath = opath.replace('.mp4', '.gif')
            print('  [!] MP4 failed, falling back to GIF: %s' % gpath)
            imageio.mimsave(gpath, images, format='GIF', duration=duration_ms, loop=0)
            opath = gpath
    return opath

def create_comparison_figure(ig, fg, gt, opath):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (title, grid) in zip(axes, [
        ('Pre-treatment (Initial)', ig),
        ('Simulated (Final)', fg),
        ('Post-treatment (Ground Truth)', gt),
    ]):
        ax.imshow(grid, cmap=CMAP, norm=NORM, interpolation='nearest')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        tf, imf = grid_fractions(grid)
        ax.legend(handles=[
            Patch(facecolor='#ffcc99', label='Tumor (%.3f)' % tf),
            Patch(facecolor='#66b3ff', label='Immune (%.3f)' % imf),
        ], loc='upper right', fontsize='small', framealpha=0.7)
    fig.tight_layout()
    fig.savefig(opath, dpi=150, bbox_inches='tight')
    plt.close(fig)

def create_rdf_evolution(grids, labels, opath, gt=None):
    n = len(grids)
    if n <= 10:
        sel = list(range(n))
    else:
        step = max(1, n // 10)
        sel = list(range(0, n, step))
        if sel[-1] != n - 1:
            sel.append(n - 1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sel)))
    for ii, gi in enumerate(sel):
        rdf, r = calculate_rdf_np(grids[gi])
        lb = labels[gi][:40] + '...' if len(labels[gi]) >= 40 else labels[gi]
        ax.plot(r, rdf, color=colors[ii], linewidth=1.5, label=lb, alpha=0.8)
    if gt is not None:
        gr, gr_r = calculate_rdf_np(gt)
        ax.plot(gr_r, gr, 'k--', lw=2.5, label='Ground Truth (post)', alpha=0.9)
    ax.set_xlabel('r (distance from center)', fontsize=12)
    ax.set_ylabel('RDF(r)', fontsize=12)
    ax.set_title('RDF Evolution Through Substeps', fontsize=14, fontweight='bold')
    ax.legend(fontsize='small', framealpha=0.7, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(opath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_fraction_evolution(grids, labels, opath, gt=None):
    n = len(grids)
    tf = np.zeros(n)
    imf = np.zeros(n)
    for i in range(n):
        tf[i], imf[i] = grid_fractions(grids[i])
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(n)
    ax.plot(x, tf, 'o-', color='#ffcc99', lw=2.5, ms=6, label='Tumor Fraction')
    ax.plot(x, imf, 's-', color='#66b3ff', lw=2.5, ms=6, label='Immune Fraction')
    if gt is not None:
        gt_t, gt_i = grid_fractions(gt)
        ax.axhline(y=gt_t, color='#ffcc99', ls='--', lw=1.5, alpha=0.6,
                   label='GT Tumor (%.3f)' % gt_t)
        ax.axhline(y=gt_i, color='#66b3ff', ls='--', lw=1.5, alpha=0.6,
                   label='GT Immune (%.3f)' % gt_i)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Substep', fontsize=12)
    ax.set_ylabel('Cell Fraction', fontsize=12)
    ax.set_title('Cell Fraction Evolution Through Substeps', fontsize=14, fontweight='bold')
    ax.legend(fontsize='small', framealpha=0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(opath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_snapshot_grid(grids, labels, opath, n_cols=4):
    n = len(grids)
    nr = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(nr, n_cols, figsize=(4 * n_cols, 4 * nr))
    axes = np.atleast_1d(axes).flatten()
    for idx in range(n):
        ax = axes[idx]
        ax.imshow(grids[idx], cmap=CMAP, norm=NORM, interpolation='nearest')
        tf, imf = grid_fractions(grids[idx])
        sl = labels[idx][:27] + '...' if len(labels[idx]) >= 30 else labels[idx]
        ax.set_title('%s\nT:%.3f IM:%.3f' % (sl, tf, imf), fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    for idx in range(n, len(axes)):
        axes[idx].axis('off')
    fig.tight_layout()
    fig.savefig(opath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_visualization(config_path, output_dir='visualizations',
                      anim_format='gif', fps=1.0, override_steps=None,
                      seed=None):
    config_path = os.path.abspath(config_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print('[0/5] Setting random seed: %d' % seed)

    print('[1/5] Reading config: %s' % config_path)
    config = read_config(config_path)
    registry = TUIM_registry()
    if override_steps is not None:
        config['simulation_metadata']['num_steps_per_episode'] = override_steps
        print('      Overrode num_steps_per_episode -> %d' % override_steps)

    print('[2/5] Initializing runner...')
    runner = TU_IM_Runner(config, registry)
    torch.device(runner.config['simulation_metadata']['device'])
    runner.init()

    print('[3/5] Running simulation...')
    with torch.no_grad():
        runner.execute()

    print('[4/5] Extracting trajectory grids...')
    grids, labels, snames = extract_trajectory_grids(runner)
    print('      Captured %d frames' % len(grids))

    ig, fg = grids[0], grids[-1]
    gt = load_ground_truth(config, config_path)
    if gt is not None:
        print('      Ground truth loaded (shape: %s)' % str(gt.shape))
    else:
        print('      No ground-truth post-treatment data found')

    print('[5/5] Producing visualizations...')

    apath = os.path.join(output_dir, 'tumor_immune_dynamics.%s' % anim_format)
    create_substep_animation(grids, labels, apath, fps=fps, fmt=anim_format)
    print('      Animation: %s' % apath)

    if gt is not None:
        cpath = os.path.join(output_dir, 'comparison_pre_sim_post.png')
        create_comparison_figure(ig, fg, gt, cpath)
        print('      Comparison: %s' % cpath)

    rpath = os.path.join(output_dir, 'rdf_evolution.png')
    create_rdf_evolution(grids, labels, rpath, gt)
    print('      RDF evolution: %s' % rpath)

    fpath = os.path.join(output_dir, 'fraction_evolution.png')
    create_fraction_evolution(grids, labels, fpath, gt)
    print('      Fraction evolution: %s' % fpath)

    spath = os.path.join(output_dir, 'substep_snapshots.png')
    create_snapshot_grid(grids, labels, spath)
    print('      Snapshots: %s' % spath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for ax, grid, title in [(ax1, ig, 'Initial (Pre-treatment)'),
                              (ax2, fg, 'Final (Simulated)')]:
        ax.imshow(grid, cmap=CMAP, norm=NORM, interpolation='nearest')
        ax.set_title(title, fontsize=13, fontweight='bold')
        t, im = grid_fractions(grid)
        ax.legend(handles=[
            Patch(facecolor='#ffcc99', label='Tumor (%.3f)' % t),
            Patch(facecolor='#66b3ff', label='Immune (%.3f)' % im),
        ], loc='upper right', fontsize='small', framealpha=0.7)
    fig.tight_layout()
    ipath = os.path.join(output_dir, 'initial_vs_final.png')
    fig.savefig(ipath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('      Initial vs Final: %s' % ipath)

    print('\n' + '=' * 60)
    print('  All visualizations saved to: %s' % output_dir)
    print('=' * 60)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Visualize tumor-immune ABM dynamics.')
    p.add_argument('--config', '-c', type=str, default='config.yaml')
    p.add_argument('--output_dir', '-o', type=str, default='visualizations')
    p.add_argument('--format', '-f', type=str, default='gif', choices=['gif', 'mp4'])
    p.add_argument('--fps', type=float, default=1.0)
    p.add_argument('--num_steps', type=int, default=None)
    p.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = p.parse_args()

    sd = os.path.dirname(os.path.abspath(__file__))
    cp = args.config if os.path.isabs(args.config) else os.path.join(sd, args.config)
    run_visualization(cp, args.output_dir, args.format, args.fps, args.num_steps, args.seed)


# """
# visualize_trajectory.py
# ========================
# Standalone script for capturing and visualizing tumor-immune ABM dynamics.

# Captures the state trajectory at each substep using the existing
# TU_IM_Runner and TUIM_registry, then produces:

#   1. Substep-level animation (GIF / MP4) with labels & cell fractions
#   2. Side-by-side comparison with post-treatment ground truth
#   3. RDF evolution over the trajectory
#   4. Cell fraction evolution over substeps
#   5. Per-substep qualitative grid snapshots (PNG)

# USAGE
# -----
#   python visualize_trajectory.py --config <path_to_config.yaml>

# This script does NOT modify any existing files in the repository.
# """

# import sys, os, argparse
# import numpy as np
# import torch
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from matplotlib.colors import ListedColormap, BoundaryNorm
# import imageio

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# sys.path.insert(0, PROJECT_ROOT)
# from simulator import TU_IM_Runner, TUIM_registry
# from AgentTorch.helpers import read_config

# CMAP = ListedColormap(['white', '#ffcc99', '#66b3ff'])
# NORM = BoundaryNorm([0, 1, 2, 3], CMAP.N)

# SUBSTEP_DISPLAY = {
#     'tumor_cell_proliferation': 'Tumor Proliferation',
#     'tumor_cell_migration': 'Tumor Migration',
#     'immune_cell_proliferation': 'Immune Proliferation',
#     'immune_cell_migration': 'Immune Migration',
#     'tumor_immune_cell_interaction': 'Combat',
#     'tumor_cell_death': 'Tumor Death',
#     'immune_cell_death': 'Immune Death',
#     'immune_cell_influx': 'Immune Influx',   # CHANGED: added
# }


# def state_to_grid(state, device='cpu'):
#     tl = state['agents']['tumorcells']['TU_location_matrix']
#     il = state['agents']['immunecells']['IM_location_matrix']
#     t2d = torch.sum(tl, dim=0)
#     i2d = torch.sum(il, dim=0)
#     g = torch.zeros_like(t2d, device=device, dtype=torch.int)
#     g[t2d > 0] = 1
#     g[i2d > 0] = 2
#     return g


# def extract_trajectory_grids(runner):
#     device = runner.config['simulation_metadata'].get('device', 'cpu')
#     snames = [runner.config['substeps'][sk]['name'] for sk in runner.config['substeps'].keys()]
#     traj = runner.state_trajectory
#     grids, labels = [], []
#     for ep_states in traj:
#         for sidx, state in enumerate(ep_states):
#             grids.append(state_to_grid(state, device=device).cpu().numpy())
#             if sidx == 0:
#                 labels.append('Initial (pre-treatment)')
#             else:
#                 d = SUBSTEP_DISPLAY.get(snames[sidx - 1], snames[sidx - 1])
#                 labels.append('After: %s' % d)
#     return grids, labels, snames


# def grid_fractions(grid):
#     t = np.sum(grid == 1) / grid.size
#     i = np.sum(grid == 2) / grid.size
#     return t, i


# def calculate_rdf_np(grid):
#     rows, cols = grid.shape
#     cy, cx = rows / 2.0, cols / 2.0
#     r = np.linspace(0, np.sqrt(rows**2 + cols**2) / 2.0, 100)
#     pos = np.argwhere(grid != 0).astype(float)
#     if len(pos) == 0:
#         return np.zeros_like(r), r
#     dists = np.sqrt(np.sum((pos - np.array([cy, cx]))**2, axis=1))
#     rdf = np.zeros_like(r)
#     for i in range(len(r)):
#         if i == 0:
#             mask = dists <= r[i]
#         else:
#             mask = (dists > r[i-1]) & (dists <= r[i])
#         rdf[i] = np.sum(mask)
#     area = np.pi * (r**2 - np.concatenate(([0], r[:-1]))**2)
#     area = np.maximum(area, 1e-12)
#     rdf = rdf / area / (np.mean(rdf[-10:]) + 1e-12)
#     return rdf, r


# def load_ground_truth(config, config_path, dl='post_10percent'):
#     """Load ground-truth post-treatment grid from CSVs in the same patient dir."""
#     tum_rel = config['simulation_metadata'].get('Tum_dense', '')
#     if not tum_rel:
#         return None
#     # Resolve tum_path to absolute using the config file location
#     cfg_dir = os.path.dirname(os.path.abspath(config_path))
#     tum_abs = os.path.abspath(os.path.join(cfg_dir, tum_rel))
#     # Go up from pre_10percent dir to patient dir, then into post_10percent
#     pre_dir = os.path.dirname(tum_abs)         # .../data/p10/pre_10percent
#     patient_dir = os.path.dirname(pre_dir)     # .../data/p10
#     post_dir = os.path.join(patient_dir, dl)   # .../data/p10/post_10percent
#     tc = os.path.join(post_dir, 'Tum_dense.csv')
#     ic = os.path.join(post_dir, 'CD8_dense.csv')
#     if not os.path.isfile(tc) or not os.path.isfile(ic):
#         print('      Ground truth CSVs not found at: %s' % post_dir)
#         return None
#     import pandas as pd
#     N, M = config['simulation_metadata']['N'], config['simulation_metadata']['M']
#     tum = pd.read_csv(tc, header=None).values.reshape(N, M)
#     imm = pd.read_csv(ic, header=None).values.reshape(N, M)
#     gt = np.zeros((N, M), dtype=int)
#     gt[tum > 0] = 1
#     gt[imm > 0] = 2
#     print('      Loaded ground truth from: %s' % post_dir)
#     return gt


# def create_substep_animation(grids, labels, opath, fps=1.0, fmt='gif'):
#     """Create an animation from grid states. fps -> duration(ms) conversion."""
#     fig, ax = plt.subplots(1, 1, figsize=(7, 7))
#     images = []
#     for idx in range(len(grids)):
#         ax.clear()
#         ax.imshow(grids[idx], cmap=CMAP, norm=NORM, interpolation='nearest')
#         tf, imf = grid_fractions(grids[idx])
#         ax.set_title('%s\nTumor: %.3f  |  Immune: %.3f' % (labels[idx], tf, imf),
#                      fontsize=11, fontweight='bold')
#         ax.set_xlabel('Column')
#         ax.set_ylabel('Row')
#         ax.legend(handles=[
#             Patch(facecolor='#ffcc99', label='Tumor (%.2f)' % tf),
#             Patch(facecolor='#66b3ff', label='Immune (%.2f)' % imf),
#         ], loc='upper right', fontsize='small', framealpha=0.7)
#         fig.tight_layout()
#         fig.canvas.draw()
#         img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#         img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         images.append(img)
#     plt.close(fig)
    
#     duration_ms = 1000.0 / fps if fps > 0 else 500
    
#     if fmt == 'gif':
#         imageio.mimsave(opath, images, format='GIF', duration=duration_ms, loop=0)
#     else:
#         try:
#             imageio.mimsave(opath, images, format='FFMPEG', fps=fps)
#         except Exception:
#             gpath = opath.replace('.mp4', '.gif')
#             print('  [!] MP4 failed, falling back to GIF: %s' % gpath)
#             imageio.mimsave(gpath, images, format='GIF', duration=duration_ms, loop=0)
#             opath = gpath
#     return opath


# def create_comparison_figure(ig, fg, gt, opath):
#     fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
#     for ax, (title, grid) in zip(axes, [
#         ('Pre-treatment (Initial)', ig),
#         ('Simulated (Final)', fg),
#         ('Post-treatment (Ground Truth)', gt),
#     ]):
#         ax.imshow(grid, cmap=CMAP, norm=NORM, interpolation='nearest')
#         ax.set_title(title, fontsize=13, fontweight='bold')
#         ax.set_xlabel('Column')
#         ax.set_ylabel('Row')
#         tf, imf = grid_fractions(grid)
#         ax.legend(handles=[
#             Patch(facecolor='#ffcc99', label='Tumor (%.3f)' % tf),
#             Patch(facecolor='#66b3ff', label='Immune (%.3f)' % imf),
#         ], loc='upper right', fontsize='small', framealpha=0.7)
#     fig.tight_layout()
#     fig.savefig(opath, dpi=150, bbox_inches='tight')
#     plt.close(fig)


# def create_rdf_evolution(grids, labels, opath, gt=None):
#     n = len(grids)
#     if n <= 10:
#         sel = list(range(n))
#     else:
#         step = max(1, n // 10)
#         sel = list(range(0, n, step))
#         if sel[-1] != n - 1:
#             sel.append(n - 1)
#     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
#     colors = plt.cm.viridis(np.linspace(0, 0.9, len(sel)))
#     for ii, gi in enumerate(sel):
#         rdf, r = calculate_rdf_np(grids[gi])
#         lb = labels[gi][:40] + '...' if len(labels[gi]) >= 40 else labels[gi]
#         ax.plot(r, rdf, color=colors[ii], linewidth=1.5, label=lb, alpha=0.8)
#     if gt is not None:
#         gr, gr_r = calculate_rdf_np(gt)
#         ax.plot(gr_r, gr, 'k--', lw=2.5, label='Ground Truth (post)', alpha=0.9)
#     ax.set_xlabel('r (distance from center)', fontsize=12)
#     ax.set_ylabel('RDF(r)', fontsize=12)
#     ax.set_title('RDF Evolution Through Substeps', fontsize=14, fontweight='bold')
#     ax.legend(fontsize='small', framealpha=0.7, loc='upper right')
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#     fig.savefig(opath, dpi=150, bbox_inches='tight')
#     plt.close(fig)


# def create_fraction_evolution(grids, labels, opath, gt=None):
#     n = len(grids)
#     tf = np.zeros(n)
#     imf = np.zeros(n)
#     for i in range(n):
#         tf[i], imf[i] = grid_fractions(grids[i])
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#     x = np.arange(n)
#     ax.plot(x, tf, 'o-', color='#ffcc99', lw=2.5, ms=6, label='Tumor Fraction')
#     ax.plot(x, imf, 's-', color='#66b3ff', lw=2.5, ms=6, label='Immune Fraction')
#     if gt is not None:
#         gt_t, gt_i = grid_fractions(gt)
#         ax.axhline(y=gt_t, color='#ffcc99', ls='--', lw=1.5, alpha=0.6,
#                    label='GT Tumor (%.3f)' % gt_t)
#         ax.axhline(y=gt_i, color='#66b3ff', ls='--', lw=1.5, alpha=0.6,
#                    label='GT Immune (%.3f)' % gt_i)
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
#     ax.set_xlabel('Substep', fontsize=12)
#     ax.set_ylabel('Cell Fraction', fontsize=12)
#     ax.set_title('Cell Fraction Evolution Through Substeps', fontsize=14, fontweight='bold')
#     ax.legend(fontsize='small', framealpha=0.7)
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#     fig.savefig(opath, dpi=150, bbox_inches='tight')
#     plt.close(fig)


# def create_snapshot_grid(grids, labels, opath, n_cols=4):
#     n = len(grids)
#     nr = (n + n_cols - 1) // n_cols
#     fig, axes = plt.subplots(nr, n_cols, figsize=(4 * n_cols, 4 * nr))
#     axes = axes.flatten() if n > 1 else [axes]
#     for idx in range(n):
#         ax = axes[idx]
#         ax.imshow(grids[idx], cmap=CMAP, norm=NORM, interpolation='nearest')
#         tf, imf = grid_fractions(grids[idx])
#         sl = labels[idx][:27] + '...' if len(labels[idx]) >= 30 else labels[idx]
#         ax.set_title('%s\nT:%.3f IM:%.3f' % (sl, tf, imf), fontsize=9, fontweight='bold')
#         ax.set_xticks([])
#         ax.set_yticks([])
#     for idx in range(n, len(axes)):
#         axes[idx].axis('off')
#     fig.tight_layout()
#     fig.savefig(opath, dpi=150, bbox_inches='tight')
#     plt.close(fig)


# def run_visualization(config_path, output_dir='visualizations',
#                       anim_format='gif', fps=1.0, override_steps=None,
#                       seed=None):
#     config_path = os.path.abspath(config_path)
#     output_dir = os.path.abspath(output_dir)
#     os.makedirs(output_dir, exist_ok=True)

#     if seed is not None:
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         print('[0/5] Setting random seed: %d' % seed)

#     print('[1/5] Reading config: %s' % config_path)
#     config = read_config(config_path)
#     registry = TUIM_registry()
#     if override_steps is not None:
#         config['simulation_metadata']['num_steps_per_episode'] = override_steps
#         print('      Overrode num_steps_per_episode -> %d' % override_steps)

#     print('[2/5] Initializing runner...')
#     runner = TU_IM_Runner(config, registry)
#     torch.device(runner.config['simulation_metadata']['device'])
#     runner.init()

#     print('[3/5] Running simulation...')
#     with torch.no_grad():
#         runner.execute()

#     print('[4/5] Extracting trajectory grids...')
#     grids, labels, snames = extract_trajectory_grids(runner)
#     print('      Captured %d frames' % len(grids))

#     ig, fg = grids[0], grids[-1]
#     gt = load_ground_truth(config, config_path)
#     if gt is not None:
#         print('      Ground truth loaded (shape: %s)' % str(gt.shape))
#     else:
#         print('      No ground-truth post-treatment data found')

#     print('[5/5] Producing visualizations...')

#     apath = os.path.join(output_dir, 'tumor_immune_dynamics.%s' % anim_format)
#     create_substep_animation(grids, labels, apath, fps=fps, fmt=anim_format)
#     print('      Animation: %s' % apath)

#     if gt is not None:
#         cpath = os.path.join(output_dir, 'comparison_pre_sim_post.png')
#         create_comparison_figure(ig, fg, gt, cpath)
#         print('      Comparison: %s' % cpath)

#     rpath = os.path.join(output_dir, 'rdf_evolution.png')
#     create_rdf_evolution(grids, labels, rpath, gt)
#     print('      RDF evolution: %s' % rpath)

#     fpath = os.path.join(output_dir, 'fraction_evolution.png')
#     create_fraction_evolution(grids, labels, fpath, gt)
#     print('      Fraction evolution: %s' % fpath)

#     spath = os.path.join(output_dir, 'substep_snapshots.png')
#     create_snapshot_grid(grids, labels, spath)
#     print('      Snapshots: %s' % spath)

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#     for ax, grid, title in [(ax1, ig, 'Initial (Pre-treatment)'),
#                               (ax2, fg, 'Final (Simulated)')]:
#         ax.imshow(grid, cmap=CMAP, norm=NORM, interpolation='nearest')
#         ax.set_title(title, fontsize=13, fontweight='bold')
#         t, im = grid_fractions(grid)
#         ax.legend(handles=[
#             Patch(facecolor='#ffcc99', label='Tumor (%.3f)' % t),
#             Patch(facecolor='#66b3ff', label='Immune (%.3f)' % im),
#         ], loc='upper right', fontsize='small', framealpha=0.7)
#     fig.tight_layout()
#     ipath = os.path.join(output_dir, 'initial_vs_final.png')
#     fig.savefig(ipath, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print('      Initial vs Final: %s' % ipath)

#     print('\n' + '=' * 60)
#     print('  All visualizations saved to: %s' % output_dir)
#     print('=' * 60)


# if __name__ == '__main__':
#     p = argparse.ArgumentParser(description='Visualize tumor-immune ABM dynamics.')
#     p.add_argument('--config', '-c', type=str, default='config.yaml')
#     p.add_argument('--output_dir', '-o', type=str, default='visualizations')
#     p.add_argument('--format', '-f', type=str, default='gif', choices=['gif', 'mp4'])
#     p.add_argument('--fps', type=float, default=1.0)
#     p.add_argument('--num_steps', type=int, default=None)
#     p.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
#     args = p.parse_args()

#     sd = os.path.dirname(os.path.abspath(__file__))
#     cp = args.config if os.path.isabs(args.config) else os.path.join(sd, args.config)
#     run_visualization(cp, args.output_dir, args.format, args.fps, args.num_steps, args.seed)
