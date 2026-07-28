import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from AgentTorch.substep import SubstepObservation
from AgentTorch.helpers import get_by_path

class GetFromState(SubstepObservation):
    """
        Get the input variables from the state
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state):
        input_variables = self.input_variables
        return {ix: get_by_path(state, re.split("/", input_variables[ix])) for ix in input_variables.keys()}

# Create Adjacency matrix here which is passed to the observation neighborhood function

class ObserveNeighborhood(SubstepObservation):
    """
        To observe the neighborhood of the tumor cells
        using the current location of the tumor cell and occupied locations

        Steps:
            1. The Neighborhood function requires the location matrix of the tumor cells
            2. This location matrix is the initial location of the tumor cells
            3. The immune location matrix is the location matrix of the immune cells
            4. We then create an occupied location map by combining the immune and tumor location maps
            5. This occupied location map is the map of the locations that are occupied by the immune and tumor cells
            6. The positions on the occupied location map cannot be occupied by the tumor cells
            7. We find the neighborhood of the tumor cells by using the convolution operation of kernel 3x3
            8. The output is the neighborhood of the tumor cells
            9. To ensure the tumor cells do not occupy the immune cells and the previously occupied locations,
            we take the and operation of the neighborhood and the not of occupied location map
            10. This provides the neighborhood of the tumor cells

        Parameters:
            immune_location_matrix (tensor): the location matrix of the immune cells
            tumor_location_matrix (tensor): the location matrix of the tumor cells

        Args:
            config (dict): the initial state
            input_variables (dict): dictionary of input variables
            output_variables (dict): dictionary of output variables
            arguments (dict): dictionary of arguments

        Returns:
            state (dict): the updated state

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state):
        input_variables = self.input_variables
        immune_location_matrix = get_by_path(state, re.split("/", input_variables["immune_location_matrix"]))
        tumor_location_matrix = get_by_path(state, re.split("/", input_variables["tumor_location_matrix"]))

        immune_location_map = torch.zeros(100, 100)
        for idx in range(immune_location_matrix.shape[0]):
            immune_location_map += immune_location_matrix[idx]
        
        tumor_location_map = torch.zeros(100, 100)
        for idx in range(tumor_location_matrix.shape[0]):
            tumor_location_map += tumor_location_matrix[idx]

        num_agents, grid_height, grid_width = tumor_location_matrix.shape
        agent_adjacency_matrices = torch.zeros((num_agents, grid_height, grid_width), dtype=torch.float32)

        kernel = torch.tensor([ [1., 1., 1.],
                                [1., 0., 1.],
                                [1., 1., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        agent_adjacency_matrices = F.conv2d(tumor_location_matrix.unsqueeze(1), kernel, padding=1).squeeze(1)

        occupied_location_map = torch.logical_or(immune_location_map, tumor_location_map).float()
        # CHANGED: actually apply the filter (previous loop reassigned a local var and did nothing)
        not_occupied = torch.logical_not(occupied_location_map).float()
        agent_adjacency_matrices = agent_adjacency_matrices * not_occupied.unsqueeze(0)

        # print("tumor cell adjacency matrix step complete!")
        return {self.output_variables[0]: agent_adjacency_matrices}