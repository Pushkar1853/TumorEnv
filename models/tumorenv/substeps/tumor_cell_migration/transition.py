import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np

class MigrateTUCell(SubstepTransition):
    """
        Transition to migrate the tumor cells to a new location
        Update the location matrices by altering positions of the cells (migration)
        This function performs the transition of the tumor cell agents to a new location

        Steps:
            1. Feed the input state to the transition function which consists of the location matrix,
            the maximum interaction capacity of the tumor cells, the maximum proliferation capacity of the tumor cells,
            and the migrate action probabilities
            2. Read the parameters from state as variables from the action substep
            3. We then have the original location matrix and the new location matrix
            4. The rule is to move the tumor cell to the new location
            5. So in the new location matrix, we will have a 1 at the new location and 0 at the old location
            6. We change the TUprolmax and TUintmax values at the new location to the maximum values
            7. We change the TUprolmax and TUintmax values at the old location to 0
            8. Update the state by altering positions of the cells by migration      

        Parameters:
            location_matrix (tensor): the location matrix of the tumor cells (initial location)  
            migrate_action (tensor): the neighborhood matrix of the tumor cells consisting of empty locations   
            TUintmax (tensor): the maximum interaction capacity of the tumor cells
            TUprolmax (tensor): the maximum proliferation capacity of the tumor cells    

        Args:
            config (dict): the input state
            input_variables (dict): dictionary of input variables
            output_variables (dict): dictionary of output variables
            arguments (dict): dictionary of arguments

        Returns:
            dict: the output state

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, state, action):
        location_matrix = get_by_path(state, re.split("/", self.input_variables["location_matrix"]))
        TUintmax = get_by_path(state, re.split("/", self.input_variables["TUintmax"]))
        TUprolmax = get_by_path(state, re.split("/", self.input_variables["TUprolmax"]))
        migrate_action = action['tumorcells']["migrate_action"]
        TUMOR_INTERACTION_CAPACITY = 2
        TUMOR_PROLIFERATION_CAPACITY = 10

        num_agents, grid_height, grid_width = location_matrix.shape

        def migrate_cell(matrix, migrate_action_probs):
            old_y, old_x = torch.where(matrix == 1)
            if len(old_x) > 1:
                # CHANGED: defensive guard — an agent should never own >1 live cell;
                # if it does (upstream bug), just take the first and log it loudly
                # instead of crashing on .item().
                print(f"[WARNING] agent has {len(old_x)} live cells, expected 1 — using first")
                old_y, old_x = old_y[:1], old_x[:1]
            if len(old_x) == 0:
                # CHANGED: dead cell (already removed) — nothing to migrate, stay all-zero
                zeros = torch.zeros_like(matrix)
                return zeros, zeros, zeros
            new_y, new_x = torch.where(migrate_action_probs > 0.5)
            new_coords = torch.stack((new_y, new_x), dim=-1)
            softmax_probs = F.softmax(migrate_action_probs[new_y, new_x], dim=0)
            if softmax_probs.size(0) > 0:
                new_coord_idx = softmax_probs.multinomial(1, False) # Choose a single cell probabilistically
                new_y, new_x = new_coords[new_coord_idx, 0], new_coords[new_coord_idx, 1]
            else:
                new_y, new_x = torch.where(matrix == 1)   # CHANGED: match (y, x) order used elsewhere
            transition_matrix = torch.zeros(migrate_action_probs.shape)
            transition_matrix[new_y, new_x] = 1
            # if len(old_x) > 1:
            #     #randomly sample one location to migrate using multinomial distribution
            #     old_coords = torch.stack((old_y, old_x), dim=-1)
            #     old_softmax_probs = F.softmax(migrate_action_probs[old_y, old_x], dim=0)
            #     old_coord_idx = old_softmax_probs.multinomial(1, False)
            #     old_tu_y, old_tu_x = old_coords[old_coord_idx, 0], old_coords[old_coord_idx, 1]
            #     transition_matrix[old_tu_x, old_tu_y] = 1
            new_matrix = transition_matrix     # original matrix is not modified
            new_Tu_prolmax = transition_matrix * TUMOR_PROLIFERATION_CAPACITY
            new_Tu_intmax = transition_matrix * TUMOR_INTERACTION_CAPACITY

            return new_matrix, new_Tu_intmax, new_Tu_prolmax
        
        updated_location_matrix = []
        updated_TUintmax = []
        updated_TUprolmax = []

        for agent_idx in range(num_agents):
            new_matrix, new_TUintmax, new_TUprolmax = migrate_cell(
                location_matrix[agent_idx], 
                migrate_action[agent_idx])
            
            updated_location_matrix.append(new_matrix)
            updated_TUintmax.append(new_TUintmax)
            updated_TUprolmax.append(new_TUprolmax)

        updated_location_matrix = torch.stack(updated_location_matrix)
        updated_TUintmax = torch.stack(updated_TUintmax)
        updated_TUprolmax = torch.stack(updated_TUprolmax)

        updated_location_matrix = updated_location_matrix.view(-1, grid_height, grid_width)
        updated_TUintmax = updated_TUintmax.view(-1, grid_height, grid_width)
        updated_TUprolmax = updated_TUprolmax.view(-1, grid_height, grid_width)

        # self.plot_location_matrices(location_matrix, updated_location_matrix)

        # print("tumor cell migration transition complete!")
        return {self.output_variables[0]: updated_location_matrix,
                self.output_variables[1]: updated_TUintmax,
                self.output_variables[2]: updated_TUprolmax}