import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import matplotlib.pyplot as plt

class KillTUCell(SubstepTransition):
    """
        Transition to kill tumor cells 
        Update the state of the tumor cells and remove dead cells from the grid

        Steps:
            1. Feed the input state with tumor location matrix, the probabilities, and the parameters
            2. Make the stack of the probabilities by get_nature function
            3. The function then follows the rule that if the probability is that agnet cell is less than 
            the death probability, then the agent cell is removed from the grid (set to zero)
            4. And if the maximum capacity of interaction or proliferation of tumor cells is less than zero,
            then also the agent cell is removed
            5. We use Gumbel softmax function to make the hard classifier differentiable
            6. Return the updated state
            
        Parameters:
            TUpdeath (float): probability of death
            TUpprol (float): probability of proliferation
            TUintmax (float): maximum capacity of interaction of tumor cells
            TUprolmax (float): maximum capacity of proliferating tumor cells
            Tum_proliferation_status (torch.tensor): proliferation status of tumor cells
            Tum_engagement_status (torch.tensor): engagement status of tumor cells
            location matrix (tensor): the location matrix of the tumor cells (original location to be removed)

        Args:
            config (dict): the initial state
            input_variables (dict): input variables
            output_variables (dict): output variables
            arguments (dict): arguments

        Returns:
            updated location matrix : the updated location matrix of the tumor cells
            
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

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def _get_nature(self, TUpmig, TUpdeath, TUpprol):
        return torch.stack([TUpprol, TUpdeath, TUpmig])

    def forward(self, state, action=None, tau=0.5):
        input_variables = self.input_variables
  
        TUpdeath = get_by_path(state, re.split("/", input_variables["TUpdeath"]))
        TUpprol = get_by_path(state, re.split("/", input_variables["TUpprol"]))
        TUpmig = get_by_path(state, re.split("/", input_variables["TUpmig"]))
        TUintmax = get_by_path(state, re.split("/", input_variables["TUintmax"]))
        TUprolmax = get_by_path(state, re.split("/", input_variables["TUprolmax"]))
        Tum_proliferation_status = get_by_path(state, re.split("/", input_variables["Tum_proliferation_status"]))
        Tum_engagement_status = get_by_path(state, re.split("/", input_variables["Tum_engagement_status"]))
        location_matrix = get_by_path(state, re.split("/", input_variables["location_matrix"]))
        TUMOR_INTERACTION_CAPACITY = 2
        TUMOR_PROLIFERATION_CAPACITY = 10

        # print(location_matrix.shape)
        num_agents, grid_height, grid_width = location_matrix.shape
        # ipdb.set_trace()

        def kill_cell(agent_matrix, TUintmax, TUprolmax):
            """
    #             => Function to kill tumor cells (remove the cells) if any of them happens:
    #                 * If the nature of the transition is death, remove all tumor cells from the grid
    #                 * If the maximum capacity of interaction of tumor cells is less than zero
    #                 * If the maximum capacity of proliferation of tumor cells is less than zero
                
    #             => If the nature is not death, return the original matrix

    #             Args:
    #                 agent_matrix (torch.tensor): the location matrix of the tumor cells
    #                 kill_x (int): x coordinate of the tumor cell to be removed from the grid
    #                 kill_y (int): y coordinate of the tumor cell to be removed from the grid
    #       """
            logits = self._get_nature(TUpmig, TUpdeath, TUpprol)
            gumbel_softmax_sample = F.gumbel_softmax(logits, tau=tau, dim=0, hard=True)
            death_weight = gumbel_softmax_sample[1]  # 1.0 if "death" was the sampled nature this step

            capacity_exhausted = torch.logical_or(TUintmax.squeeze() <= 0, TUprolmax.squeeze() <= 0).squeeze()
            # CHANGED: dies now also fires when the stochastic nature draw is "death",
            # not only when capacity happens to hit zero. Previously death_weight was
            # computed and discarded, so TUpdeath had no effect on the simulation.
            dies = torch.logical_or(capacity_exhausted, death_weight.bool())

            survives = (~dies).float()
            transition_matrix = agent_matrix * survives
            new_matrix = transition_matrix
            # CHANGED: preserve each surviving cell's existing capacity instead of
            # resetting it to a constant every step (which erased accumulated combat damage)
            new_TU_intmax = TUintmax * survives
            new_TU_prolmax = TUprolmax * survives
            new_Tu_proliferation_status = torch.zeros(transition_matrix.shape)
            new_Tu_engagement_status = torch.zeros(transition_matrix.shape)
            return new_matrix, new_TU_intmax, new_TU_prolmax, new_Tu_proliferation_status, new_Tu_engagement_status

        updated_agent_matrix = []
        updated_TUintmax = []
        updated_TUprolmax = []
        updated_Tum_proliferation_status = []
        updated_Tum_engagement_status = []

        for agent_idx in range(num_agents):
            new_matrix, new_TUintmax, new_TUprolmax, new_Tum_prolif_status, new_Tum_engage_status = kill_cell(
                location_matrix[agent_idx],
                TUintmax[agent_idx],
                TUprolmax[agent_idx])
            updated_agent_matrix.append(new_matrix)
            updated_TUintmax.append(new_TUintmax)
            updated_TUprolmax.append(new_TUprolmax)
            updated_Tum_proliferation_status.append(new_Tum_prolif_status)
            updated_Tum_engagement_status.append(new_Tum_engage_status)

        # Stack the tensors and ensure correct shape
        updated_agent_matrix = torch.stack(updated_agent_matrix)
        updated_TUintmax = torch.stack(updated_TUintmax)
        updated_TUprolmax = torch.stack(updated_TUprolmax)
        updated_Tum_proliferation_status = torch.stack(updated_Tum_proliferation_status)
        updated_Tum_engagement_status = torch.stack(updated_Tum_engagement_status)

        updated_agent_matrix = updated_agent_matrix.view(-1, grid_height, grid_width)
        updated_TUintmax = updated_TUintmax.view(-1, grid_height, grid_width)
        updated_TUprolmax = updated_TUprolmax.view(-1, grid_height, grid_width)
        updated_Tum_proliferation_status = updated_Tum_proliferation_status.view(-1, grid_height, grid_width)
        updated_Tum_engagement_status = updated_Tum_engagement_status.view(-1, grid_height, grid_width)

        # self.plot_location_matrices(location_matrix, updated_agent_matrix)

        # print("tumor cell death transition complete!")
        return {self.output_variables[0]: updated_agent_matrix,
                self.output_variables[1]: updated_TUintmax,
                self.output_variables[2]: updated_TUprolmax,
                self.output_variables[3]: updated_Tum_proliferation_status,
                self.output_variables[4]: updated_Tum_engagement_status}