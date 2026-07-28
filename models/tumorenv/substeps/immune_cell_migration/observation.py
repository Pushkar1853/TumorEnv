import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from AgentTorch.substep import SubstepObservation
from AgentTorch.helpers import get_by_path

class GetFromState(SubstepObservation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state):
        input_variables = self.input_variables
        return {ix: get_by_path(state, re.split("/", input_variables[ix])) for ix in input_variables.keys()}
    
class IM_ObserveFarNeighborhood(SubstepObservation): # This consists of chemotaxis map
    """
        Observe the far neighborhood of the immune cell
        using the current location of the immune cell and occupied locations

        Steps:
            1. The far neighborhood of the immune cell requires the location matrix of the immune cells
            2. This location matrix is the initial location of the immune cells (immune location matrix)
            3. We then create an occupied location map by combining the immune and tumor location maps
            4. This occupied location map is the map of the locations that are occupied by the immune and tumor cells
            5. The positions on the occupied location map cannot be occupied by the immune cells
            6. We find the neighborhood of the immune cells by using the convolution operation of kernel 5x5
            7. The output is the far neighborhood of the immune cells with all the empty cells possible
            8. To ensure the immune cells do not occupy the tumor cells and the previously occupied locations,
                we take the and operation of the neighborhood and the not of occupied location map
            9. This provides the far neighborhood of the immune cells
            10. For immune cells migration observation we use the Chemotaxis map
            11. This map represents the chessboard distances of each empty cell with all tumor cells present in the grid
            12. This map is used to identify the most probable location to migrate for the immune cell
            13. The final adjacency matrix passed represented the most probable location selected from the chemotaxis map
            14. This final adjacency matrix has 1 where the empty location is the most closest to tumor cells

        Parameters:
            location_matrix (tensor): the location matrix of the tumor cells
            immune_location_matrix (tensor): the location matrix of the immune cells

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
        tum_num_agents, grid_height, grid_width = tumor_location_matrix.shape

        immune_location_map = immune_location_matrix.sum(dim=0)
        tumor_location_map = tumor_location_matrix.sum(dim=0)

        num_agents, grid_height, grid_width = immune_location_matrix.shape
        final_agent_adjacency_matrices = torch.zeros((num_agents, grid_height, grid_width), dtype=torch.float32)

        kernel = torch.tensor([[1., 1., 1., 1., 1.],
                               [1., 1., 1., 1., 1.],
                               [1., 1., 0., 1., 1.],
                               [1., 1., 1., 1., 1.],
                               [1., 1., 1., 1., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        agent_adjacency_matrices = F.conv2d(immune_location_matrix.unsqueeze(1), kernel, padding=2).squeeze(1)

        # CHANGED: actually apply the occupied-cell mask (previous loop was a no-op)
        occupied_location_map = torch.logical_or(immune_location_map > 0, tumor_location_map > 0).float()
        not_occupied = torch.logical_not(occupied_location_map).float()
        agent_adjacency_matrices = agent_adjacency_matrices * not_occupied.unsqueeze(0)

        def get_distance_optimized(tumor_matrix, agent_adjacency_map):
            # CHANGED: candidates must be actual valid (unoccupied, in-neighborhood) cells,
            # not "== max()" which breaks once the map is all-zero for some agents
            empty_loc_y, empty_loc_x = torch.where(agent_adjacency_map > 0)
            tumor_loc_y, tumor_loc_x = torch.where(tumor_matrix == 1)
            if len(empty_loc_x) == 0 or len(tumor_loc_x) == 0:
                return torch.stack((torch.tensor(-1), torch.tensor(-1)))
            empty_locs = torch.stack((empty_loc_x, empty_loc_y), dim=1).unsqueeze(1)
            tumor_locs = torch.stack((tumor_loc_x, tumor_loc_y), dim=1).unsqueeze(0)
            distances = torch.max(torch.abs(empty_locs - tumor_locs), dim=2).values
            min_distances, _ = torch.min(distances, dim=1)
            min_dist_idx = torch.argmin(min_distances)  # CHANGED: nearest, not farthest
            min_location = torch.stack((empty_loc_x[min_dist_idx], empty_loc_y[min_dist_idx]))
            return min_location

        # CHANGED: claim cells as we go so two immune cells can't target the same site this step
        claimed_map = occupied_location_map.clone()

        for agent_idx in range(num_agents):
            available = agent_adjacency_matrices[agent_idx] * (1 - claimed_map)
            min_location = get_distance_optimized(tumor_location_map, available)
            if min_location[0].item() == -1:
                continue
            final_agent_adjacency_matrices[agent_idx][min_location[0], min_location[1]] = 1
            claimed_map[min_location[0], min_location[1]] = 1

        # print("immune cell migration observation complete!")
        return {self.output_variables[0]: final_agent_adjacency_matrices}