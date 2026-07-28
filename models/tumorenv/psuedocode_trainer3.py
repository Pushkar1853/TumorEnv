import os
import gc
import torch
import numpy as np
import pandas as pd
import cma  # pip install cma
from AgentTorch.helpers import read_config, set_by_path
from simulator import TU_IM_Runner, TUIM_registry
from visualize_trajectory import calculate_rdf_np, grid_fractions, create_comparison_figure

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(SCRIPT_DIR, "config.yaml")
config = read_config(config_path)
registry = TUIM_registry()

PARAM_NAMES = ["TUpprol", "TUpmig", "TUpdeath", "TUps",
               "IMpprol", "IMpmig", "IMpdeath", "IMpkill", "IMinfluxProb"]

# Tune the CMA-ES budget here. Start small to confirm everything runs, then
# scale up (e.g. popsize=8, maxiter=30) for a real calibration result.
CMA_POPSIZE = 8
CMA_MAXITER = 3

# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------
def load_ground_truth_fixed(config, config_path):
    """Robust ground-truth loader: uses whatever filenames are in config
    instead of assuming Tum_dense.csv, and resolves post_10percent/ from
    the pre_10percent/ path already in config."""
    cfg_dir = os.path.dirname(os.path.abspath(config_path))

    tum_pre = os.path.abspath(os.path.join(cfg_dir, config["simulation_metadata"]["Tum_dense"]))
    imm_pre = os.path.abspath(os.path.join(cfg_dir, config["simulation_metadata"]["CD8_dense"]))

    pre_dir = os.path.dirname(tum_pre)
    patient_dir = os.path.dirname(pre_dir)
    post_dir = os.path.join(patient_dir, "post_10percent")

    tum_post = os.path.join(post_dir, os.path.basename(tum_pre))
    imm_post = os.path.join(post_dir, os.path.basename(imm_pre))

    print("=" * 80)
    print("Loading ground truth")
    print("Tumor :", tum_post)
    print("Immune:", imm_post)
    print("=" * 80)

    if not os.path.isfile(tum_post):
        raise FileNotFoundError(tum_post)
    if not os.path.isfile(imm_post):
        raise FileNotFoundError(imm_post)

    N = config["simulation_metadata"]["N"]
    M = config["simulation_metadata"]["M"]

    tum = pd.read_csv(tum_post, header=None).values.reshape(N, M)
    imm = pd.read_csv(imm_post, header=None).values.reshape(N, M)

    gt = np.zeros((N, M), dtype=np.int32)
    gt[tum > 0] = 1
    gt[imm > 0] = 2
    return gt

# ---------------------------------------------------------------------------
# Loss: fraction + RDF + local clustering
# ---------------------------------------------------------------------------
import torch.nn.functional as F_nn

_CLUSTER_KERNEL = torch.tensor([[1., 1., 1.],
                                 [1., 0., 1.],
                                 [1., 1., 1.]], dtype=torch.float32).view(1, 1, 3, 3)

def local_clustering_score(grid: np.ndarray, cell_type: int):
    """Fraction of same-type neighbors, averaged over all occupied cells of
    that type. ~0 = scattered/speckled, ~1 = solid clumps. Not center-relative
    (unlike RDF), so it catches local aggregation structure RDF alone misses."""
    mask = (grid == cell_type).astype(np.float32)
    if mask.sum() == 0:
        return 0.0
    mask_t = torch.from_numpy(mask).view(1, 1, *mask.shape)
    same_type_neighbor_count = F_nn.conv2d(mask_t, _CLUSTER_KERNEL, padding=1).squeeze().numpy()
    occupied = mask > 0
    neighbor_fraction = same_type_neighbor_count[occupied] / 8.0
    return float(neighbor_fraction.mean())

def simulation_loss(sim_grid, gt_grid: np.ndarray, gt_rdf, gt_r,
                     rdf_weight=0.3, cluster_weight=0.3, return_components=False,
                     rdf_skip_bins=2):  # CHANGED: skip the first few bins near r=0
    sim_grid_np = sim_grid.numpy() if isinstance(sim_grid, torch.Tensor) else sim_grid

    sim_t_frac, sim_i_frac = grid_fractions(sim_grid_np)
    gt_t_frac, gt_i_frac = grid_fractions(gt_grid)
    fraction_loss = (sim_t_frac - gt_t_frac) ** 2 + (sim_i_frac - gt_i_frac) ** 2

    sim_rdf, sim_r = calculate_rdf_np(sim_grid_np)
    gt_rdf_interp = np.interp(sim_r, gt_r, gt_rdf)

    # CHANGED: drop the first `rdf_skip_bins` bins entirely — near r=0 the
    # annulus area is ~0, so a single centered cell produces a divide-by-
    # near-zero blowup (rdf value ~1e10+) that swamps every other signal
    # in the loss. Real structural information lives in the mid/outer bins.
    valid = slice(rdf_skip_bins, None)
    rdf_loss = float(np.mean((sim_rdf[valid] - gt_rdf_interp[valid]) ** 2))

    sim_tumor_cluster = local_clustering_score(sim_grid_np, cell_type=1)
    gt_tumor_cluster = local_clustering_score(gt_grid, cell_type=1)
    sim_immune_cluster = local_clustering_score(sim_grid_np, cell_type=2)
    gt_immune_cluster = local_clustering_score(gt_grid, cell_type=2)
    cluster_loss = (sim_tumor_cluster - gt_tumor_cluster) ** 2 + (sim_immune_cluster - gt_immune_cluster) ** 2

    total_loss = float(fraction_loss) + rdf_weight * rdf_loss + cluster_weight * cluster_loss

    if return_components:
        return {
            "total": total_loss, "fraction_loss": float(fraction_loss),
            "rdf_loss": rdf_loss, "cluster_loss": float(cluster_loss),
            "sim_tumor_frac": sim_t_frac, "gt_tumor_frac": gt_t_frac,
            "sim_immune_frac": sim_i_frac, "gt_immune_frac": gt_i_frac,
            "sim_tumor_cluster": sim_tumor_cluster, "gt_tumor_cluster": gt_tumor_cluster,
            "sim_immune_cluster": sim_immune_cluster, "gt_immune_cluster": gt_immune_cluster,
        }
    return total_loss

# ---------------------------------------------------------------------------
# Parameter application
# ---------------------------------------------------------------------------
def apply_params_to_state(runner, params: dict):
    """params: {'TUpprol': 0.5, 'TUpmig': 0.3, ...} — plain floats."""
    for name, value in params.items():
        path = f"environment/{name}"
        set_by_path(root=runner.state, items=path.split("/"), value=torch.tensor(float(value)))

def to_prob(x):
    return 1 / (1 + np.exp(-x))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ground_truth_grid = load_ground_truth_fixed(config, config_path)
    gt_rdf, gt_r = calculate_rdf_np(ground_truth_grid)

    def objective(x):
        params = {name: to_prob(xi) for name, xi in zip(PARAM_NAMES, x)}

        runner = TU_IM_Runner(config, registry)
        runner.init()
        apply_params_to_state(runner, params)

        with torch.no_grad():
            runner.step(config['simulation_metadata']['num_steps_per_episode'])

        sim_grid = runner.get_current_state()
        loss = simulation_loss(sim_grid, ground_truth_grid, gt_rdf, gt_r)

        del runner
        gc.collect()
        return loss

    # Seed CMA-ES at the config's current default values (inverse-sigmoid)
    x0 = [np.log(config['simulation_metadata'][name] / (1 - config['simulation_metadata'][name]))
          for name in PARAM_NAMES]

    es = cma.CMAEvolutionStrategy(x0, 0.5, {'popsize': CMA_POPSIZE, 'maxiter': CMA_MAXITER})
    loss_history = []

    print(f"\nStarting CMA-ES calibration: popsize={CMA_POPSIZE}, maxiter={CMA_MAXITER}")
    print(f"Total forward rollouts: ~{CMA_POPSIZE * CMA_MAXITER}\n")

    while not es.stop():
        solutions = es.ask()
        losses = [objective(x) for x in solutions]
        es.tell(solutions, losses)
        loss_history.append(min(losses))
        print(f"Generation {len(loss_history)}: best loss={min(losses):.5f}")

    best_x = es.result.xbest
    calibrated_params = {name: to_prob(xi) for name, xi in zip(PARAM_NAMES, best_x)}

    print("\n" + "=" * 60)
    print("Calibrated parameters:")
    for k, v in calibrated_params.items():
        default_v = config['simulation_metadata'][k]
        print(f"  {k:15s}  default={default_v:.4f}  calibrated={v:.4f}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Final rerun with calibrated params — capture initial (pre-treatment)
    # grid BEFORE applying params/stepping, so the comparison figure is real.
    # -----------------------------------------------------------------------
    runner = TU_IM_Runner(config, registry)
    runner.init()
    initial_sim_grid = runner.get_current_state()   # real pre-treatment grid
    apply_params_to_state(runner, calibrated_params)
    with torch.no_grad():
        runner.step(config['simulation_metadata']['num_steps_per_episode'])
    final_sim_grid = runner.get_current_state()

    os.makedirs(os.path.join(SCRIPT_DIR, "visualizations"), exist_ok=True)
    create_comparison_figure(
        ig=initial_sim_grid.numpy(),
        fg=final_sim_grid.numpy(),
        gt=ground_truth_grid,
        opath=os.path.join(SCRIPT_DIR, "visualizations", "final_calibrated_comparison.png"),
    )

    loss_breakdown = simulation_loss(final_sim_grid, ground_truth_grid, gt_rdf, gt_r, return_components=True)
    pre_t, pre_i = grid_fractions(initial_sim_grid.numpy())

    print("\n" + "=" * 60)
    print("Final loss breakdown:")
    print(f"  Total loss:          {loss_breakdown['total']:.5f}")
    print(f"  Fraction loss:       {loss_breakdown['fraction_loss']:.5f}")
    print(f"  RDF loss:            {loss_breakdown['rdf_loss']:.5f}")
    print(f"  Clustering loss:     {loss_breakdown['cluster_loss']:.5f}")
    print("-" * 60)
    print(f"  Tumor fraction   — pre: {pre_t:.3f}  sim: {loss_breakdown['sim_tumor_frac']:.3f}  gt: {loss_breakdown['gt_tumor_frac']:.3f}")
    print(f"  Immune fraction  — pre: {pre_i:.3f}  sim: {loss_breakdown['sim_immune_frac']:.3f}  gt: {loss_breakdown['gt_immune_frac']:.3f}")
    print(f"  Tumor clustering — sim: {loss_breakdown['sim_tumor_cluster']:.3f}  gt: {loss_breakdown['gt_tumor_cluster']:.3f}")
    print(f"  Immune clustering— sim: {loss_breakdown['sim_immune_cluster']:.3f}  gt: {loss_breakdown['gt_immune_cluster']:.3f}")
    print("=" * 60)
    print(f"\nSaved comparison figure to: visualizations/final_calibrated_comparison.png")


if __name__ == "__main__":
    main()