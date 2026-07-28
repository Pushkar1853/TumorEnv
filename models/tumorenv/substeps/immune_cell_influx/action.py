import re
import torch
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import get_by_path


class InfluxIMDecision(SubstepAction):
    """
        For every currently-dead immune cell agent slot, independently decide
        with probability IMinfluxProb whether a fresh immune cell enters the
        tissue this step, and pick a candidate grid location for it uniformly
        among valid empty sites near the tumor.

        Parameters:
            IMinfluxProb (float): per-step probability a dead slot gets replenished

        Returns:
            influx_targets (tensor): [num_agents, H, W], 1 at the chosen new-cell
                position for slots that influx this step, all-zero otherwise
    """
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state, observation):
        input_variables = self.input_variables
        IMinfluxProb = get_by_path(state, re.split("/", input_variables["IMinfluxProb"]))

        dead_mask = observation["dead_mask"]
        candidate_map = observation["candidate_map"]

        num_agents = dead_mask.shape[0]
        grid_height, grid_width = candidate_map.shape
        influx_targets = torch.zeros((num_agents, grid_height, grid_width), dtype=torch.float32)

        candidate_y, candidate_x = torch.where(candidate_map > 0)
        num_candidates = candidate_y.shape[0]
        dead_indices = torch.where(dead_mask)[0]

        if num_candidates > 0 and dead_indices.shape[0] > 0:
            influx_roll = torch.rand(dead_indices.shape[0])
            influx_happens = influx_roll < IMinfluxProb

            for i in range(dead_indices.shape[0]):
                if influx_happens[i]:
                    agent_idx = dead_indices[i]
                    pick = torch.randint(0, num_candidates, (1,)).item()
                    y, x = candidate_y[pick], candidate_x[pick]
                    influx_targets[agent_idx, y, x] = 1
                    # CHANGED: remove this site so two agents can't influx onto the same cell this step
                    remaining = torch.ones(num_candidates, dtype=torch.bool)
                    remaining[pick] = False
                    candidate_y, candidate_x = candidate_y[remaining], candidate_x[remaining]
                    num_candidates = candidate_y.shape[0]
                    if num_candidates == 0:
                        break

        # print("immune cell influx action complete!")
        return {self.output_variables[0]: influx_targets}