import numpy as np
import re
import torch
from torch.nn import functional as F
from AgentTorch.substep import SubstepObservation
from AgentTorch.helpers import get_by_path
import matplotlib.pyplot as plt
import ipdb

"""
    Logic:
    1. Checking the chemotaxis map to identify immune cells close to tumor cells.
    2. Identifying eligible immune cells based on their current state.
    3. Examining the neighborhood of immune cells to find potential tumor cell targets.
    4. Identifying tumor cells in the neighborhood.
"""

class GetFromState(SubstepObservation):
    """
        Get the variables from the state
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state):
        input_variables = self.input_variables
        return {ix: get_by_path(state, re.split("/", input_variables[ix])) for ix in input_variables.keys()}

class ObserveIMkNeighborhood(SubstepObservation):
    """
        This function checks the chemotaxis map to identify immune cells close to tumor cells.
        It then identifies eligible immune cells based on their current state.
        Finally, it examines the neighborhood of immune cells to find potential tumor cell targets.

        Args:
            config (dict): the initial state
            input_variables (dict): dictionary of input variables
            output_variables (dict): dictionary of output variables
            arguments (dict): dictionary of arguments

        Returns:
            eligible_immune_cells (tensor): the eligible immune cells
            neighborhood_tumor_matrices (tensor): the neighborhood tumor matrices

        Parameters:
            immune_location_matrix (tensor): the location matrix of the immune cells
            tumor_location_matrix (tensor): the location matrix of the tumor cells

        Steps:
            1. Feed the initial state to the observation function
            2. Read the parameters from state as variables from the observation substep
            3. Get the adjacent locations of the immune cells to the tumor cells using the chessboard distance 
            4. If the adjacent location list is not empty, then the probable location matrix is updated
            5. Identify the eligible immune cells based on their current state
            6. Get the neighborhood of immune cells to find potential tumor cell targets
            7. Return the eligible immune cells and the neighborhood tumor matrices
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adjacency_kernel = torch.nn.Parameter(torch.tensor([[1., 1., 1.],
                                                                 [1., 0., 1.],
                                                                 [1., 1., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.chemotaxis_threshold = torch.nn.Parameter(torch.tensor(1.0))

    # Get the adjacent locations of the immune cells to the tumor cells using the chessboard distance

    def get_chemotaxis_map(self, tumor_location_matrix, immune_location_matrix):
        immune_location_map = torch.zeros(100, 100)
        for idx in range(immune_location_matrix.shape[0]):
            immune_location_map += immune_location_matrix[idx]

        tumor_location_map = torch.zeros(100, 100)
        for idx in range(tumor_location_matrix.shape[0]):
            tumor_location_map += tumor_location_matrix[idx]
        agent_adjacency_matrices = F.conv2d(tumor_location_matrix.unsqueeze(1), self.adjacency_kernel, padding=1).squeeze(1)
        occupied_location_map = torch.logical_or(immune_location_map, tumor_location_map).float()
        for tumor_cell_map in agent_adjacency_matrices:
            tumor_cell_map = torch.logical_and(tumor_cell_map, torch.logical_not(occupied_location_map)).float()
        return agent_adjacency_matrices
    # Get the eligible immune cells based on their current state

    def get_nonzero_indices(self, tensor):
        return torch.nonzero(tensor > 0.99, as_tuple=False)

    def forward(self, state):
        input_variables = self.input_variables

        immune_location_matrix = get_by_path(state, re.split("/", input_variables["immune_location_matrix"]))  # Shape: [N_immune_cells, H, W]
        tumor_location_matrix = get_by_path(state, re.split("/", input_variables["location_matrix"]))  # Shape: [N_tumor_cells, H, W]
        IMkmax = get_by_path(state, re.split("/", input_variables["IMkmax"]))
        CD8_engagement_status = get_by_path(state, re.split("/", input_variables["CD8_engagement_status"]))

        neighborhood_tumor_matrices = F.conv2d(
            tumor_location_matrix.unsqueeze(1), self.adjacency_kernel, padding=1
        ).squeeze(1)  # [N_tumor_cells, H, W]

        eligible_immune_cells = {}
        # Iterate over each tumor cell
        for tumor_cell_idx in range(tumor_location_matrix.shape[0]):
            # Get the neighborhood matrix for this tumor cell
            neighborhood_tumor_matrix = neighborhood_tumor_matrices[tumor_cell_idx]

            # Initialize the list for this tumor cell index
            eligible_immune_cells[tumor_cell_idx] = []

            # For each immune cell, check if it is in the neighborhood
            for immune_cell_idx in range(immune_location_matrix.shape[0]):
                # Get the location of the immune cell (non-zero indices)
                immune_non_zero_indices = self.get_nonzero_indices(immune_location_matrix[immune_cell_idx])

                # CHANGED: guard against a dead/removed immune cell with no nonzero location
                if immune_non_zero_indices.shape[0] == 0:
                    continue

                x, y = immune_non_zero_indices[0][0], immune_non_zero_indices[0][1]

                # Check if the immune cell is in the neighborhood of the tumor cell
                if neighborhood_tumor_matrix[x, y] > 0:
                    # The immune cell is in the neighborhood; add to the list
                    eligible_immune_cells[tumor_cell_idx].append((immune_cell_idx, x, y))

        # CHANGED: dedented out of the for-loop so ALL tumor cells get processed, not just index 0
        # total_eligible_pairs = sum(len(v) for v in eligible_immune_cells.values())
        # print(f"[DEBUG] {total_eligible_pairs} eligible immune-tumor adjacent pairs found this step")
        # print("Combat Observation completed!")
        return {self.output_variables[0]: eligible_immune_cells,
                self.output_variables[1]: neighborhood_tumor_matrices,
                self.output_variables[2]: IMkmax,
                self.output_variables[3]: CD8_engagement_status}