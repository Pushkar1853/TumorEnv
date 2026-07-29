"""
Gradient verification script for the differentiable shadow population accumulator.

This script:
1. Sets up the simulation with learnable parameters (as nn.Parameters)
2. Runs a single step
3. Calls loss.backward() on the soft population fractions
4. Checks that gradients flow back to the learnable parameters
5. Prints gradient magnitudes for each parameter
"""
import sys
sys.path.insert(0, '../../')

import torch
import torch.nn as nn
import re
from simulator import TU_IM_Runner, TUIM_registry
from AgentTorch.helpers import read_config, get_by_path, set_by_path

print("=" * 70)
print("GRADIENT VERIFICATION TEST")
print("=" * 70)

# Load config
config = read_config("config.yaml")
registry = TUIM_registry()

# Override to run just 1 step for quick testing
config['simulation_metadata']['num_steps_per_episode'] = 1
config['simulation_metadata']['num_episodes'] = 1

# Create runner
runner = TU_IM_Runner(config, registry)
runner.init()

# Define the 9 calibration parameters as nn.Parameters
# We use logit-space parameters so they stay in (0,1) after sigmoid
PARAM_NAMES = ["TUpprol", "TUpmig", "TUpdeath", "TUps",
               "IMpprol", "IMpmig", "IMpdeath", "IMpkill", "IMinfluxProb"]

# Get initial values from config
raw_params = nn.ParameterDict()
for name in PARAM_NAMES:
    init_val = torch.tensor(config['simulation_metadata'][name])
    # Store in logit space so sigmoid gives (0,1)
    logit_val = torch.logit(init_val, eps=1e-6)
    raw_params[name] = nn.Parameter(logit_val)

print("\nInitial parameter values:")
for name, p in raw_params.items():
    print(f"  {name}: {torch.sigmoid(p).item():.6f}")

# Set parameters into state
for name, raw in raw_params.items():
    set_by_path(root=runner.state, items=["environment", name], value=torch.sigmoid(raw))

# Run one step WITHOUT torch.no_grad()
print("\nRunning simulation step (with autograd enabled)...")
try:
    runner.step(config['simulation_metadata']['num_steps_per_episode'])
    print("  Step completed successfully!")
except Exception as e:
    print(f"  ERROR during step: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get soft population fractions
print("\nComputing soft population fractions...")
try:
    pred_tumor_frac, pred_immune_frac = runner.get_soft_population_fractions()
    print(f"  Predicted tumor fraction: {pred_tumor_frac.item():.6f}")
    print(f"  Predicted immune fraction: {pred_immune_frac.item():.6f}")
    print(f"  Requires grad (tumor): {pred_tumor_frac.requires_grad}")
    print(f"  Requires grad (immune): {pred_immune_frac.requires_grad}")
    print(f"  Grad fn (tumor): {pred_tumor_frac.grad_fn}")
    print(f"  Grad fn (immune): {pred_immune_frac.grad_fn}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compute loss
gt_tumor_frac = 0.102
gt_immune_frac = 0.207
loss = (pred_tumor_frac - gt_tumor_frac) ** 2 + (pred_immune_frac - gt_immune_frac) ** 2
print(f"\nLoss: {loss.item():.6f}")
print(f"Loss requires grad: {loss.requires_grad}")
print(f"Loss grad_fn: {loss.grad_fn}")

# Backward
print("\n" + "=" * 70)
print("BACKWARD PASS")
print("=" * 70)
try:
    loss.backward()
    print("  backward() completed successfully!")
except Exception as e:
    print(f"  ERROR during backward: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check gradients
print("\n" + "=" * 70)
print("GRADIENT RESULTS")
print("=" * 70)
print(f"{'Parameter':<20} {'Value':<12} {'Gradient':<12} {'|Grad|>0':<10}")
print("-" * 54)

has_any_grad = False
for name, raw in raw_params.items():
    grad = raw.grad
    grad_norm = grad.norm().item() if grad is not None else 0.0
    has_grad = grad is not None and grad_norm > 0
    if has_grad:
        has_any_grad = True
    print(f"{name:<20} {torch.sigmoid(raw).item():<12.6f} {grad_norm:<12.8f} {'YES' if has_grad else 'NO':<10}")

print("-" * 54)
if has_any_grad:
    print("SUCCESS: Gradients are flowing through the differentiable shadow accumulator!")
    print("   The loss.backward() successfully propagated gradients from")
    print("   population fractions back to the calibration parameters.")
else:
    print("FAILURE: No gradients detected!")
    print("   The gradient chain is broken somewhere.")

# Also check the soft_delta values directly
print("\n" + "=" * 70)
print("SOFT DELTA STATE VALUES")
print("=" * 70)
soft_tumor_delta = get_by_path(runner.state, ["environment", "soft_tumor_delta"])
soft_immune_delta = get_by_path(runner.state, ["environment", "soft_immune_delta"])
print(f"  soft_tumor_delta: {soft_tumor_delta.item():.6f} (requires_grad={soft_tumor_delta.requires_grad})")
print(f"  soft_immune_delta: {soft_immune_delta.item():.6f} (requires_grad={soft_immune_delta.requires_grad})")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)