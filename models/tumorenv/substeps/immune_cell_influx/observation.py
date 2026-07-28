import numpy as np
import re
import torch
import torch.nn.functional as F
from AgentTorch.substep import SubstepObservation
from AgentTorch.helpers import get_by_path


class GetFromState(SubstepObservation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state):
        input_variables = self.input_variables
        return {ix: get_by_path(state, re.split("/", input_variables[ix])) for ix in input_variables.keys()}


class ObserveInfluxCandidates(SubstepObservation):
    """
        Find which immune-cell agent slots are currently dead (all-zero
        location matrix) and are therefore eligible to be replenished, and
        compute a map of empty grid cells near the tumor mass where a fresh
        immune cell could plausibly enter from the vasculature this step.

        Returns:
            dead_mask (tensor): [num_agents] bool, True where the agent slot is empty
            candidate_map (tensor): [H, W] float, 1 at valid empty entry sites near tumor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adjacency_kernel = torch.nn.Parameter(torch.tensor(
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 0., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self, state):
        input_variables = self.input_variables
        immune_location_matrix = get_by_path(state, re.split("/", input_variables["immune_location_matrix"]))
        tumor_location_matrix = get_by_path(state, re.split("/", input_variables["tumor_location_matrix"]))

        dead_mask = (immune_location_matrix.sum(dim=(1, 2)) == 0)

        immune_location_map = immune_location_matrix.sum(dim=0)
        tumor_location_map = tumor_location_matrix.sum(dim=0)
        occupied_map = torch.logical_or(immune_location_map > 0, tumor_location_map > 0)

        tumor_proximity = F.conv2d(
            tumor_location_map.unsqueeze(0).unsqueeze(0), self.adjacency_kernel, padding=2
        ).squeeze()
        candidate_map = torch.logical_and(tumor_proximity > 0, torch.logical_not(occupied_map)).float()

        # print("influx candidate observation complete!")
        return {self.output_variables[0]: dead_mask,
                self.output_variables[1]: candidate_map}