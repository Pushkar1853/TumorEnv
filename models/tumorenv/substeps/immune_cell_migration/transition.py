import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import matplotlib.pyplot as plt

class MigrateIMCell(SubstepTransition):
    """
        Transition for the immune cells to migrate
        This transition is used to migrate the immune cells to the empty locations
        This function performs the transition of the immune cell agents to a new location

        Steps:
            1. Feed the input state with the location matrix, the maximum interaction capacity of the immune cells,
            the maximum interaction capacity of the immune cells, the maximum killing capacity of the immune cells,
            the maximum proliferation capacity of the immune cells, and the migrate action probabilities
            2. Read the parameters from state as variables from the action substep
            3. We then have the original location matrix and the new location matrix
            4. The rule is to move the immune cell to the new location
            5. So in the new location matrix, we will have a 1 at the new location and 0 at the old location
            6. We change the IMprolmax, IMintmax, IMkmax values at the new location to the maximum values
            7. We change the IMprolmax, IMintmax, IMkmax values at the old location to 0
            8. Update the state by altering positions of the cells by migration

        Parameters:
            location_matrix (tensor): the location matrix of the immune cells (initial location)  
            migrate_action (tensor): the neighborhood matrix of the immune cells consisting of empty locations   
            IMprolmax (tensor): the maximum proliferating capacity of the immune cells
            IMintmax (tensor): the maximum interaction capacity of the immune cells
            IMkmax (tensor): the maximum killing capacity of the immune cells
            engagementDuration (tensor): the engagement duration assigned to the immune cells
            migrate_action (tensor): the new neighborhood of the immune cells (new location)

        Args:
            config (dict): the input state
            input_variables (dict): dictionary of input variables
            output_variables (dict): dictionary of output variables
            arguments (dict): dictionary of arguments

        Returns:
            state (dict): the output state

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def forward(self, state, action):
        input_variables = self.input_variables
        location_matrix = get_by_path(state, re.split("/", input_variables["immune_location_matrix"]))
        IMprolmax = get_by_path(state, re.split("/", input_variables["IMprolmax"]))
        IMintmax = get_by_path(state, re.split("/", input_variables["IMintmax"]))
        IMkmax = get_by_path(state, re.split("/", input_variables["IMkmax"]))
        engagementDuration = get_by_path(state, re.split("/", input_variables["engagementDuration"]))
        migrate_action = action['immunecells']["migrate_action"]
        IMMUNE_PROLIFERATION_CAPACITY = 10
        IMMUNE_INTERACTION_CAPACITY = 40
        IMMUNE_KILLING_CAPACITY = 5
        ENGAGEMENT_DURATION = 48

        num_agents, grid_height, grid_width = location_matrix.shape

        # CHANGED: track cells claimed by agents already processed this substep so
        # two agents can't migrate onto the same cell within one step — previously
        # that "erased" one of them from the pixel count even though both still
        # existed in the underlying per-agent tensors.
        claimed_map = (location_matrix.sum(dim=0) > 0).float()

        def migrate_cell(matrix, migrate_action_probs, IMprolmax, IMintmax, IMkmax, engagementDuration, claimed_map):
            old_y, old_x = torch.where(matrix == 1)
            if len(old_x) > 1:
                # CHANGED: defensive guard — an agent should never own >1 live cell;
                # if it does (upstream bug), just take the first and log it loudly
                # instead of crashing on .item().
                print(f"[WARNING] agent has {len(old_x)} live cells, expected 1 — using first")
                old_y, old_x = old_y[:1], old_x[:1]
            if len(old_x) == 0:
                zeros = torch.zeros_like(matrix)
                return zeros, zeros, zeros, zeros, zeros, None

            # CHANGED: exclude own current cell and anything claimed this step already
            available_probs = migrate_action_probs * (1 - claimed_map)
            available_probs[old_y, old_x] = 0

            new_y, new_x = torch.where(available_probs > 0.5)
            new_coords = torch.stack((new_y, new_x), dim=-1)
            softmax_probs = F.softmax(available_probs[new_y, new_x], dim=0)
            if softmax_probs.size(0) > 0:
                new_coord_idx = softmax_probs.multinomial(1, False)
                new_y, new_x = new_coords[new_coord_idx, 0], new_coords[new_coord_idx, 1]
                claim = (new_y.item(), new_x.item())
            else:
                # CHANGED: no free candidate this step -> stay put
                new_y, new_x = old_y, old_x
                claim = (old_y.item(), old_x.item())

            transition_matrix = torch.zeros(migrate_action_probs.shape)
            transition_matrix[new_y, new_x] = 1

            new_matrix = transition_matrix
            new_IMprolmax = transition_matrix * IMMUNE_PROLIFERATION_CAPACITY
            new_IMintmax = transition_matrix * IMMUNE_INTERACTION_CAPACITY
            new_IMkmax = transition_matrix * IMMUNE_KILLING_CAPACITY
            new_engagementDuration = transition_matrix * ENGAGEMENT_DURATION

            return new_matrix, new_IMprolmax, new_IMintmax, new_IMkmax, new_engagementDuration, claim

        updated_location_matrix, updated_IMprolmax, updated_IMintmax = [], [], []
        updated_IMkmax, updated_engagementDuration = [], []

        for agent_idx in range(num_agents):
            new_matrix, new_IMprolmax, new_IMintmax, new_IMkmax, new_engagementDuration, claim = migrate_cell(
                location_matrix[agent_idx], migrate_action[agent_idx],
                IMprolmax[agent_idx], IMintmax[agent_idx], IMkmax[agent_idx], engagementDuration[agent_idx],
                claimed_map
            )
            if claim is not None:
                claimed_map[claim[0], claim[1]] = 1  # CHANGED: reserve for the rest of this step's agents

            updated_location_matrix.append(new_matrix)
            updated_IMprolmax.append(new_IMprolmax)
            updated_IMintmax.append(new_IMintmax)
            updated_IMkmax.append(new_IMkmax)
            updated_engagementDuration.append(new_engagementDuration)

        updated_location_matrix = torch.stack(updated_location_matrix).view(-1, grid_height, grid_width)
        updated_IMprolmax = torch.stack(updated_IMprolmax).view(-1, grid_height, grid_width)
        updated_IMintmax = torch.stack(updated_IMintmax).view(-1, grid_height, grid_width)
        updated_IMkmax = torch.stack(updated_IMkmax).view(-1, grid_height, grid_width)
        updated_engagementDuration = torch.stack(updated_engagementDuration).view(-1, grid_height, grid_width)

        # self.plot_location_matrices(location_matrix, updated_location_matrix)

        # print("immune cell migration transition complete!")
        return {self.output_variables[0]: updated_location_matrix,
                self.output_variables[1]: updated_IMprolmax,
                self.output_variables[2]: updated_IMintmax,
                self.output_variables[3]: updated_IMkmax,
                self.output_variables[4]: updated_engagementDuration}