import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import random
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import matplotlib.pyplot as plt

class ProliferateIMCell(SubstepTransition):
    """
        Transition to proliferate the Immune cells
        This function updates the parameters of the immune cells based on the proliferate_cell_index
        The empty location gets a cell added, and the parameters are subsequently updated of that location

        Steps:
            1. Feed the state to the transition function which consists of the location matrix,
            the maximum interaction capacity of the immune cells, the maximum proliferation capacity of the immune cells,
            the probability of proliferation, the proliferation status of the immune cells, the engagement status of the immune cells,
            and the neighborhood location matrix via policy step
            2. Read the variable values from the state and store them in the variables
            3. We update the parameters based on the empty locations on the neighborhood matrix:
                (given by the action substep)
                a. If the empty index is the index of the parent immune cells, (The original location)
                    we update the parameters of the immune cells by decreasing the capacity parameter
                b. If the empty index is the index of the new daughter immune cells, (The new location)
                    we update the parameters of the immune cells by keeping the capacity parameter same
                Rest of the params are same in both the cases
            4. Update the state by altering the positions of new locations to 1 and changing the values
                of the parameters of the immune cells

        Parameters:
            immune_location_matrix (torch.tensor): the location matrix of the immune cells (original location)
            IMprolmax (torch.tensor): maximum capacity of proliferation
            IMintmax (torch.tensor): maximum number of interactions
            CD8_proliferation_status (torch.tensor): proliferation status of the immune cells
            CD8_non_proliferation_status (torch.tensor): non-proliferation status of the immune cells
            CD8_engagement_status (torch.tensor): engagement status of the immune cells
            new_neighborhood (torch.tensor): the neighborhood of the immune cells (new locations)

        Args:
            config (dict): The input state
            input_variables (dict): The input variables
            output_variables (dict): The output variables
            arguments (dict): The arguments dictionary
        
        Returns:
            state (dict): The updated state

    """
    # def plot_location_matrices(self, initial_matrix, updated_matrix):
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    #     # Sum over all agents
    #     initial_sum = (initial_matrix.sum(dim=0)>0).cpu().numpy()
    #     updated_sum = (updated_matrix.sum(dim=0)>0).cpu().numpy()

    #     # Plot initial matrix
    #     im1 = ax1.imshow(initial_sum, cmap='coolwarm', interpolation='nearest')
    #     ax1.set_title('Initial Location Matrix (Sum)')
    #     plt.colorbar(im1, ax=ax1, label='Cell Count')

    #     # Plot updated matrix
    #     im2 = ax2.imshow(updated_sum, cmap='coolwarm', interpolation='nearest')
    #     ax2.set_title('Updated Location Matrix (Sum)')
    #     plt.colorbar(im2, ax=ax2, label='Cell Count')

    #     # Set common labels
    #     for ax in (ax1, ax2):
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Y')

    #     plt.tight_layout()
    #     plt.savefig('location_matrices.png')
    #     plt.show()
    #     plt.close()

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state, action):
        input_variables = self.input_variables
        location_matrix = get_by_path(state, re.split("/", input_variables["immune_location_matrix"]))
        IMprolmax = get_by_path(state, re.split("/", input_variables["IMprolmax"]))
        IMintmax = get_by_path(state, re.split("/", input_variables["IMintmax"]))
        CD8_proliferation_status = get_by_path(state, re.split("/", input_variables["CD8_proliferation_status"]))
        CD8_non_proliferation_status = get_by_path(state, re.split("/", input_variables["CD8_non_proliferation_status"]))
        CD8_engagement_status = get_by_path(state, re.split("/", input_variables["CD8_engagement_status"]))
        prolif_action = action['immunecells']['prolif_action']
        IMMUNE_PROLIFERATION_CAPACITY = 10
        IMMUNE_INTERACTION_CAPACITY = 40

        num_agents, grid_height, grid_width = location_matrix.shape

        updated_location_matrix = location_matrix.clone()
        updated_IMprolmax = IMprolmax.clone()
        updated_IMintmax = IMintmax.clone()
        updated_CD8_proliferation_status = CD8_proliferation_status.clone()
        updated_CD8_non_proliferation_status = CD8_non_proliferation_status.clone()
        updated_CD8_engagement_status = CD8_engagement_status.clone()

        dead_slot_mask = (location_matrix.sum(dim=(1, 2)) == 0)
        dead_slot_indices = torch.where(dead_slot_mask)[0].tolist()
        dead_slot_ptr = 0

        claimed_map = (location_matrix.sum(dim=0) > 0).float()

        for agent_idx in range(num_agents):
            if dead_slot_mask[agent_idx]:
                continue

            matrix = location_matrix[agent_idx]
            probs = prolif_action[agent_idx]

            old_y, old_x = torch.where(matrix == 1)
            if len(old_x) == 0:
                continue

            candidate_probs = probs.clone()
            candidate_probs[old_y, old_x] = 0
            candidate_probs = candidate_probs * (1 - claimed_map)

            cand_y, cand_x = torch.where(candidate_probs > 0.5)
            if cand_y.numel() == 0:
                continue

            if dead_slot_ptr >= len(dead_slot_indices):
                continue

            softmax_probs = F.softmax(candidate_probs[cand_y, cand_x], dim=0)
            pick = softmax_probs.multinomial(1, False)
            new_y, new_x = cand_y[pick], cand_x[pick]

            daughter_idx = dead_slot_indices[dead_slot_ptr]
            dead_slot_ptr += 1

            updated_location_matrix[daughter_idx] = torch.zeros(grid_height, grid_width)
            updated_location_matrix[daughter_idx][new_y, new_x] = 1
            updated_IMprolmax[daughter_idx] = torch.zeros(grid_height, grid_width)
            updated_IMprolmax[daughter_idx][new_y, new_x] = IMMUNE_PROLIFERATION_CAPACITY
            updated_IMintmax[daughter_idx] = torch.zeros(grid_height, grid_width)
            updated_IMintmax[daughter_idx][new_y, new_x] = IMMUNE_INTERACTION_CAPACITY
            updated_CD8_proliferation_status[daughter_idx] = torch.zeros(grid_height, grid_width)
            updated_CD8_non_proliferation_status[daughter_idx] = torch.zeros(grid_height, grid_width)
            updated_CD8_non_proliferation_status[daughter_idx][new_y, new_x] = 1
            updated_CD8_engagement_status[daughter_idx] = torch.zeros(grid_height, grid_width)

            claimed_map[new_y, new_x] = 1

        # self.plot_location_matrices(location_matrix, updated_location_matrix)

        # print("immune cell proliferation transition complete! (%d new cells)" % dead_slot_ptr)
        return {self.output_variables[0]: updated_location_matrix,
                self.output_variables[1]: updated_IMprolmax,
                self.output_variables[2]: updated_CD8_proliferation_status,
                self.output_variables[3]: updated_CD8_non_proliferation_status,
                self.output_variables[4]: updated_CD8_engagement_status,
                self.output_variables[5]: updated_IMintmax}