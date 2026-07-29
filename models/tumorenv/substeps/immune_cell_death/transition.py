import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import matplotlib.pyplot as plt

class KillIMCell(SubstepTransition):
    """
        Transition to kill immune cells in the state
        Update the state of the immune cells depending upon the rules
        The rule is if even one of the conditions is true, then the immune cell dies
        and removed from the the location matrix

        Steps:
            1. Feed the input state with immune location matrix, the probabilities, and the parameters
            2. Make the stack of the probabilities by get_nature function
            3. The function then follows the rule that if the probability is that agnet cell is less than 
            the death probability, then the agent cell is removed from the grid (set to zero)
            4. And if the maximum killing capacity of immune cells is less than zero or the 
            maximum proliferation capacity of immune cells is less than zero or the maximum proliferation 
            capacity of immune cells is greater than maximum, then also the agent cell is removed
            5. We use Gumbel softmax function to make the hard classifier differentiable
            6. Return the updated state

        Parameters:
            IMprolmax (float): maximum capacity of proliferation for immune cells
            IMkmax (float): maximum killing capacity of immune cells
            IMpprol (float): probability of proliferation of immune cells
            IMpdeath (float): probability of death of immune cells
            CD8_proliferation_status (int): status of CD8 cells
            CD8_non_proliferation_status (int): status of CD8 cells
            CD8_engagement_status (int): status of CD8 cells
            location matrix (tensor): the location matrix of the immune cells (original location to be removed)

        Args:
            config (dict): the input state
            input_variables (dict): dictionary of input variables
            output_variables (dict): dictionary of output variables
            arguments (dict): dictionary of arguments

        Returns:
            The updated location matrix of the immune cells

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
    
    def _get_nature(self,IMpprol, IMpdeath, IMpmig):
        return torch.stack([IMpprol, IMpdeath, IMpmig])

    def forward(self, state, action=None, tau=0.5):
        input_variables = self.input_variables

        IMpprol = get_by_path(state, re.split("/", input_variables["IMpprol"]))
        IMpmig = get_by_path(state, re.split("/", input_variables["IMpmig"]))
        IMpdeath = get_by_path(state, re.split("/", input_variables["IMpdeath"]))
        
        IMprolmax = get_by_path(state, re.split("/", input_variables["IMprolmax"]))
        IMkmax = get_by_path(state, re.split("/", input_variables["IMkmax"]))
        CD8_proliferation_status = get_by_path(state, re.split("/", input_variables["CD8_proliferation_status"]))
        CD8_non_proliferation_status = get_by_path(state, re.split("/", input_variables["CD8_non_proliferation_status"]))
        CD8_engagement_status = get_by_path(state, re.split("/", input_variables["CD8_engagement_status"]))
        location_matrix = get_by_path(state, re.split("/", input_variables["immune_location_matrix"]))
        soft_immune_delta = get_by_path(state, re.split("/", input_variables["soft_immune_delta"]))
        IMMUNE_PROLIFERATION_CAPACITY = 10
        IMMUNE_INTERACTION_CAPACITY = 40
        IMMUNE_KILLING_CAPACITY = 5
        ENGAGEMENT_DURATION = 48

        num_agents, grid_height, grid_width = location_matrix.shape 

        def kill_cell(agent_matrix, IMkmax, IMprolmax):
            """
                => Function to kill immune cells (remove the cells) if any of these happen: 
                    * If the nature of the transition is death, remove all immune cells from the grid
                    * If the maximum killing capacity of immune cells is less than zero 
                    * If the maximum proliferation capacity of immune cells is less than zero
                    * If the maximum proliferation capacity of immune cells is greater than maximum
                
                => If the nature is not death:
                    * return the original matrix

                Args:
                    agent_matrix (torch.tensor): the location matrix of the immune cells
                    kill_x (int): x coordinate of the immune cell to be removed from the grid
                    kill_y (int): y coordinate of the immune cell to be removed from the grid

            """
            logits = self._get_nature(IMpprol, IMpdeath, IMpmig)
            gumbel_softmax_sample = F.gumbel_softmax(logits, tau=tau, dim=0, hard=True)
            death_weight = gumbel_softmax_sample[1]

            capacity_exhausted = (torch.logical_or(torch.logical_or(IMkmax == 0, IMprolmax == 0), IMprolmax > 10)).view(grid_height, grid_width)
            # CHANGED: previously a cell with capacity_exhausted=True only died if the
            # SAME timestep's random nature draw happened to land on "death" (~1/3 chance
            # each step by default), decoupling capacity depletion from actual death.
            # Now capacity exhaustion kills unconditionally; the nature draw is an
            # additional, independent chance of death (e.g. from IMpdeath probability).
            dies = torch.logical_or(capacity_exhausted, death_weight.bool())
            survives = (~dies).float()

            transition_matrix = agent_matrix * survives

            new_matrix = transition_matrix
            # CHANGED: preserve existing capacity for survivors instead of resetting
            # to a constant, so combat damage carries over between steps
            new_IM_kmax = IMkmax * survives
            new_IM_prolmax = IMprolmax * survives
            new_CD8_prol_status = torch.zeros(transition_matrix.shape)
            new_CD8_non_prol_status = torch.zeros(transition_matrix.shape)
            new_CD8_eng_status = torch.zeros(transition_matrix.shape)

            return new_matrix, new_IM_kmax, new_IM_prolmax, new_CD8_prol_status, new_CD8_non_prol_status, new_CD8_eng_status, death_weight

        updated_location_matrix = []
        updated_IMprolmax = []
        updated_IMkmax = []
        updated_CD8_proliferation_status = []
        updated_CD8_non_proliferation_status = []
        updated_CD8_engagement_status = []
        collected_death_weights = []

        # kill_x, kill_y: immune_cells_id
        for agent_idx in range(num_agents):
            new_matrix, new_IMkmax, new_IMprolmax, new_CD8_proliferation_status, new_CD8_non_proliferation_status, new_CD8_engagement_status, death_weight = kill_cell(
                location_matrix[agent_idx], 
                IMkmax[agent_idx], 
                IMprolmax[agent_idx])
        
            updated_location_matrix.append(new_matrix)
            updated_IMprolmax.append(new_IMprolmax)
            updated_IMkmax.append(new_IMkmax)
            updated_CD8_proliferation_status.append(new_CD8_proliferation_status)
            updated_CD8_non_proliferation_status.append(new_CD8_non_proliferation_status)
            updated_CD8_engagement_status.append(new_CD8_engagement_status)
            collected_death_weights.append(death_weight)

        updated_location_matrix = torch.stack(updated_location_matrix)
        updated_IMprolmax = torch.stack(updated_IMprolmax)
        updated_IMkmax = torch.stack(updated_IMkmax)
        updated_CD8_proliferation_status = torch.stack(updated_CD8_proliferation_status)
        updated_CD8_non_proliferation_status = torch.stack(updated_CD8_non_proliferation_status)
        updated_CD8_engagement_status = torch.stack(updated_CD8_engagement_status)

        updated_location_matrix = updated_location_matrix.view(-1, grid_height, grid_width)
        updated_IMprolmax = updated_IMprolmax.view(-1, grid_height, grid_width)
        updated_IMkmax = updated_IMkmax.view(-1, grid_height, grid_width)
        updated_CD8_proliferation_status = updated_CD8_proliferation_status.view(-1, grid_height, grid_width)
        updated_CD8_non_proliferation_status = updated_CD8_non_proliferation_status.view(-1, grid_height, grid_width)
        updated_CD8_engagement_status = updated_CD8_engagement_status.view(-1, grid_height, grid_width)

        # CHANGED: differentiable shadow accumulator for immune death events
        death_weights = torch.stack(collected_death_weights)
        died_mask = (updated_location_matrix.sum(dim=(1, 2)) == 0) & (location_matrix.sum(dim=(1, 2)) > 0)
        step_delta = -(died_mask.float().detach() * death_weights).sum()
        updated_soft_immune_delta = soft_immune_delta + step_delta

        # self.plot_location_matrices(location_matrix, updated_location_matrix)

        # print("immune cell death transition complete!") 
        return {self.output_variables[0]: updated_location_matrix,
                self.output_variables[1]: updated_IMkmax,
                self.output_variables[2]: updated_IMprolmax,
                self.output_variables[3]: updated_CD8_proliferation_status,
                self.output_variables[4]: updated_CD8_non_proliferation_status,
                self.output_variables[5]: updated_CD8_engagement_status,
                self.output_variables[6]: updated_soft_immune_delta}
