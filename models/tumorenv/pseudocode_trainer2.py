import torch
from AgentTorch.helpers import read_config
from simulator import TU_IM_Runner, TUIM_registry
from visualize_trajectory import load_ground_truth, calculate_rdf_np, grid_fractions

config_path = 'config.yaml'  # points at pre_10percent for initialization
config = read_config(config_path)
registry = TUIM_registry()
runner = TU_IM_Runner(config, registry)
runner.init()

calibration_nn = MyCalibNN()   # outputs TUpprol, TUpmig, TUpdeath, IMpprol, IMpmig, IMpdeath, IMpkill, IMinfluxProb, ...
optimizer = torch.optim.SGD(calibration_nn.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Ground truth target, loaded once (post-treatment biopsy)
ground_truth_grid = load_ground_truth(config, config_path)          # numpy [H, W]
gt_rdf, gt_r = calculate_rdf_np(ground_truth_grid)
gt_tumor_frac, gt_immune_frac = grid_fractions(ground_truth_grid)
gt_target = torch.tensor([gt_tumor_frac, gt_immune_frac], dtype=torch.float32)

num_episodes = 50
num_steps_per_episode = config['simulation_metadata']['num_steps_per_episode']

for episode in range(num_episodes):
    runner.reset()                     # re-seeds state from pre_10percent every episode
    optimizer.zero_grad()

    # "features" here = summary stats of the PRE-treatment grid you already have,
    # no SegNN needed since coordinates are already segmented
    initial_state_features = extract_features(runner.get_current_state())  # e.g. density map or RDF/fraction vector
    simulation_params = calibration_nn(initial_state_features)             # dict or tensor of predicted probabilities

    # write predicted params into the runner's environment state BEFORE stepping,
    # keeping them as the live tensors calibration_nn produced (no .item()/.detach())
    apply_params_to_state(runner, simulation_params)

    runner.step(num_steps_per_episode)
    final_grid = runner.get_current_state()   # torch tensor, [H, W], values in {0,1,2}

    sim_tumor_frac = (final_grid == 1).float().mean()
    sim_immune_frac = (final_grid == 2).float().mean()
    sim_target = torch.stack([sim_tumor_frac, sim_immune_frac])

    loss = loss_fn(sim_target, gt_target)
    loss.backward()
    optimizer.step()

    print(f"Episode {episode}: loss={loss.item():.4f}")