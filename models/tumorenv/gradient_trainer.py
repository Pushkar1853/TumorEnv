"""
gradient_trainer.py
====================
Gradient-based calibration of the tumor-immune ABM using the differentiable
shadow population accumulator (soft_tumor_delta / soft_immune_delta).

Companion to calibrate.py (CMA-ES). Structured identically so results are
directly comparable in the report: same ground-truth loader, same final
loss breakdown format, same comparison figure output.

IMPORTANT SCOPE NOTE: the soft accumulator only tracks population COUNT
changes from proliferation, death, and influx (via their Gumbel-softmax
event weights). It does NOT cover Combat (discrete health-attrition loop,
deliberately excluded — see design discussion) or spatial placement
(migration/positions remain hard-sampled throughout). Consequently:
  - Only 7 of 9 parameters receive gradient signal: TUpprol, TUpmig,
    TUpdeath, IMpprol, IMpmig, IMpdeath, IMinfluxProb
  - TUps (symmetric-division prob) and IMpkill (combat) get zero gradient
    and are held fixed at their default values during gradient training
  - The gradient itself is a biased/heuristic estimator (straight-through-
    style: hard "did this fire" mask, soft weight for "how much"), not an
    unbiased pathwise or REINFORCE gradient
  - The soft fraction prediction is a proxy quantity, not literally read
    off the hard grid — so we do a separate, real (no_grad) rollout with
    the final calibrated params to get the actual grid for RDF/clustering/
    visual comparison, exactly as the CMA-ES script does.
"""
import os
import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from AgentTorch.helpers import read_config, get_by_path, set_by_path
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

# Parameters confirmed (via gradient_test.py) to receive nonzero gradient
# through the soft accumulator. The other two (TUps, IMpkill) are held
# fixed at their config default throughout gradient training.
GRADIENT_ACTIVE_PARAMS = ["TUpprol", "TUpmig", "TUpdeath",
                           "IMpprol", "IMpmig", "IMpdeath", "IMinfluxProb"]
GRADIENT_FROZEN_PARAMS = ["TUps", "IMpkill"]

NUM_EPISODES = 10
LEARNING_RATE = 0.05


# ---------------------------------------------------------------------------
# Ground truth loading (identical to calibrate.py)
# ---------------------------------------------------------------------------
def load_ground_truth_fixed(config, config_path):
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    tum_pre = os.path.abspath(os.path.join(cfg_dir, config["simulation_metadata"]["Tum_dense"]))
    imm_pre = os.path.abspath(os.path.join(cfg_dir, config["simulation_metadata"]["CD8_dense"]))

    pre_dir = os.path.dirname(tum_pre)
    patient_dir = os.path.dirname(pre_dir)
    post_dir = os.path.join(patient_dir, "post_10percent")

    tum_post = os.path.join(post_dir, os.path.basename(tum_pre))
    imm_post = os.path.join(post_dir, os.path.basename(imm_pre))

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
# Loss components (identical to calibrate.py, reused for final evaluation)
# ---------------------------------------------------------------------------
import torch.nn.functional as F_nn

_CLUSTER_KERNEL = torch.tensor([[1., 1., 1.],
                                 [1., 0., 1.],
                                 [1., 1., 1.]], dtype=torch.float32).view(1, 1, 3, 3)


def local_clustering_score(grid: np.ndarray, cell_type: int):
    mask = (grid == cell_type).astype(np.float32)
    if mask.sum() == 0:
        return 0.0
    mask_t = torch.from_numpy(mask).view(1, 1, *mask.shape)
    same_type_neighbor_count = F_nn.conv2d(mask_t, _CLUSTER_KERNEL, padding=1).squeeze().numpy()
    occupied = mask > 0
    return float((same_type_neighbor_count[occupied] / 8.0).mean())


def simulation_loss(sim_grid, gt_grid, gt_rdf, gt_r, rdf_weight=0.3, cluster_weight=0.3,
                     return_components=False, rdf_skip_bins=2):
    sim_grid_np = sim_grid.numpy() if isinstance(sim_grid, torch.Tensor) else sim_grid

    sim_t_frac, sim_i_frac = grid_fractions(sim_grid_np)
    gt_t_frac, gt_i_frac = grid_fractions(gt_grid)
    fraction_loss = (sim_t_frac - gt_t_frac) ** 2 + (sim_i_frac - gt_i_frac) ** 2

    sim_rdf, sim_r = calculate_rdf_np(sim_grid_np)
    gt_rdf_interp = np.interp(sim_r, gt_r, gt_rdf)
    valid = slice(rdf_skip_bins, None)
    rdf_loss = float(np.mean((sim_rdf[valid] - gt_rdf_interp[valid]) ** 2))

    sim_tumor_cluster = local_clustering_score(sim_grid_np, 1)
    gt_tumor_cluster = local_clustering_score(gt_grid, 1)
    sim_immune_cluster = local_clustering_score(sim_grid_np, 2)
    gt_immune_cluster = local_clustering_score(gt_grid, 2)
    cluster_loss = (sim_tumor_cluster - gt_tumor_cluster) ** 2 + (sim_immune_cluster - gt_immune_cluster) ** 2

    total_loss = float(fraction_loss) + rdf_weight * rdf_loss + cluster_weight * cluster_loss

    if return_components:
        return {"total": total_loss, "fraction_loss": float(fraction_loss),
                "rdf_loss": rdf_loss, "cluster_loss": float(cluster_loss),
                "sim_tumor_frac": sim_t_frac, "gt_tumor_frac": gt_t_frac,
                "sim_immune_frac": sim_i_frac, "gt_immune_frac": gt_i_frac,
                "sim_tumor_cluster": sim_tumor_cluster, "gt_tumor_cluster": gt_tumor_cluster,
                "sim_immune_cluster": sim_immune_cluster, "gt_immune_cluster": gt_immune_cluster}
    return total_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ground_truth_grid = load_ground_truth_fixed(config, config_path)
    gt_rdf, gt_r = calculate_rdf_np(ground_truth_grid)
    gt_tumor_frac, gt_immune_frac = grid_fractions(ground_truth_grid)

    print(f"Ground truth — tumor fraction: {gt_tumor_frac:.4f}, immune fraction: {gt_immune_frac:.4f}")
    print(f"Gradient-active parameters ({len(GRADIENT_ACTIVE_PARAMS)}): {GRADIENT_ACTIVE_PARAMS}")
    print(f"Frozen (no gradient path) parameters ({len(GRADIENT_FROZEN_PARAMS)}): {GRADIENT_FROZEN_PARAMS}\n")

    # -----------------------------------------------------------------------
    # Learnable parameters — logit space so sigmoid keeps them in (0,1)
    # -----------------------------------------------------------------------
    raw_params = nn.ParameterDict()
    for name in GRADIENT_ACTIVE_PARAMS:
        init_val = torch.tensor(float(config['simulation_metadata'][name]))
        raw_params[name] = nn.Parameter(torch.logit(init_val, eps=1e-6))

    frozen_values = {name: float(config['simulation_metadata'][name]) for name in GRADIENT_FROZEN_PARAMS}

    optimizer = torch.optim.Adam(raw_params.parameters(), lr=LEARNING_RATE)

    runner = TU_IM_Runner(config, registry)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    loss_history = []
    tumor_frac_history = []
    immune_frac_history = []

    print(f"Starting gradient-based calibration: {NUM_EPISODES} episodes, lr={LEARNING_RATE}\n")

    for episode in range(NUM_EPISODES):
        optimizer.zero_grad()
        runner.reset()   # re-seeds from pre_10percent, resets soft_*_delta to 0, records initial fractions

        for name, raw in raw_params.items():
            set_by_path(root=runner.state, items=["environment", name], value=torch.sigmoid(raw))
        for name, value in frozen_values.items():
            set_by_path(root=runner.state, items=["environment", name], value=torch.tensor(value))

        # NO torch.no_grad() — this is the whole point, we need the graph retained
        runner.step(config['simulation_metadata']['num_steps_per_episode'])

        pred_tumor_frac, pred_immune_frac = runner.get_soft_population_fractions()

        loss = (pred_tumor_frac - gt_tumor_frac) ** 2 + (pred_immune_frac - gt_immune_frac) ** 2
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        tumor_frac_history.append(pred_tumor_frac.item())
        immune_frac_history.append(pred_immune_frac.item())

        if episode % 10 == 0 or episode == NUM_EPISODES - 1:
            print(f"Episode {episode:3d}: loss={loss.item():.5f}  "
                  f"pred_tumor={pred_tumor_frac.item():.3f}  pred_immune={pred_immune_frac.item():.3f}")

    calibrated_params = {name: torch.sigmoid(raw).item() for name, raw in raw_params.items()}
    calibrated_params.update(frozen_values)

    print("\n" + "=" * 60)
    print("Gradient-calibrated parameters:")
    for k in PARAM_NAMES:
        default_v = config['simulation_metadata'][k]
        status = "GRADIENT" if k in GRADIENT_ACTIVE_PARAMS else "FROZEN"
        print(f"  {k:15s}  default={default_v:.4f}  calibrated={calibrated_params[k]:.4f}  [{status}]")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Loss curve plot
    # -----------------------------------------------------------------------
    os.makedirs(os.path.join(SCRIPT_DIR, "visualizations"), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(loss_history, color='#66b3ff', lw=2)
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Loss (soft fraction MSE)")
    ax1.set_title("Gradient-Based Calibration Loss"); ax1.grid(alpha=0.3)

    ax2.plot(tumor_frac_history, label="Predicted Tumor Fraction", color='#ffcc99', lw=2)
    ax2.plot(immune_frac_history, label="Predicted Immune Fraction", color='#66b3ff', lw=2)
    ax2.axhline(gt_tumor_frac, ls='--', color='#ffcc99', alpha=0.6, label=f"GT Tumor ({gt_tumor_frac:.3f})")
    ax2.axhline(gt_immune_frac, ls='--', color='#66b3ff', alpha=0.6, label=f"GT Immune ({gt_immune_frac:.3f})")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Fraction"); ax2.legend(fontsize='small')
    ax2.set_title("Soft Fraction Convergence"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, "visualizations", "gradient_training_curves.png"), dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Final REAL (hard, no_grad) rollout with calibrated params — for the
    # actual grid, RDF, clustering, and visual comparison figure. The soft
    # accumulator is a proxy used only for backprop; evaluation against the
    # real simulator output is what belongs in the report.
    # -----------------------------------------------------------------------
    runner.reset()
    initial_sim_grid = runner.get_current_state()
    for name, value in calibrated_params.items():
        set_by_path(root=runner.state, items=["environment", name], value=torch.tensor(float(value)))
    with torch.no_grad():
        runner.step(config['simulation_metadata']['num_steps_per_episode'])
    final_sim_grid = runner.get_current_state()

    create_comparison_figure(
        ig=initial_sim_grid.numpy(),
        fg=final_sim_grid.numpy(),
        gt=ground_truth_grid,
        opath=os.path.join(SCRIPT_DIR, "visualizations", "gradient_calibrated_comparison.png"),
    )

    loss_breakdown = simulation_loss(final_sim_grid, ground_truth_grid, gt_rdf, gt_r, return_components=True)
    pre_t, pre_i = grid_fractions(initial_sim_grid.numpy())

    print("\n" + "=" * 60)
    print("Final loss breakdown (real hard rollout, calibrated params):")
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
    print(f"\nSaved: visualizations/gradient_training_curves.png")
    print(f"Saved: visualizations/gradient_calibrated_comparison.png")

    return calibrated_params, loss_breakdown, loss_history


if __name__ == "__main__":
    main()