import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import matplotlib.pyplot as plt

class ProliferateTUCell(SubstepTransition):
    """
        Transition for the proliferation of tumor cells 
        Update the parameters of the tumor cells based on the proliferation nature (symmetric or asymmetric)

        Steps:
            1. Feed the state to the transition function which consists of the location matrix,
            the maximum interaction capacity of the tumor cells, the maximum proliferation capacity of the tumor cells,
            the probability of proliferation, the proliferation status of the tumor cells, the engagement status of the tumor cells,
            the symmetric proliferation probability and neighborhood location matrix via policy step
            2. Decide the nature of the proliferation (symmetric or asymmetric) on the basis of TUps
                a. If random number is less than TUps, then symmetric proliferation
                b. Else asymmetric proliferation
            3. Read the variable values from the state and store them in the variables
            4. This classification is to be made differentiable using Softmax function
            4. According to the nature of proliferation, we update the parameters of the tumor cells
            5. These tumors cell are of two types: one parent and a new daughter cell (decided by the action substep)
                * Parent cells have original location and the new daughter cell have neighborhood location
                a. For the parent cell, we update the parameters as per the nature of proliferation
                    for both types of proliferation, we decrease the capacity of proliferation and interaction
                    by 1, and rest remains same
                b. For the daughter cell, we update the parameters
                    for symmetric proliferation, we keep the capacity of proliferation and interaction same
                    for asymmetric proliferation, we decrease the capacity of proliferation and interaction
                    by 1, and rest remains same
            6. Update the state by altering the positions of new locations to 1 and changing the values
                of the parameters of the tumor cells 

        Parameters:
            empty neighborhood obtained from observation and passed through the action substep
            The original cell location and the new cell location is passed through the action substep
            TUprolmax (torch.tensor): maximum capacity of proliferation
            TUpprol (torch.tensor): probability of proliferation
            Tum_proliferation_status (torch.tensor): proliferation status of the tumor cells
            Tum_engagement_status (torch.tensor): engagement status of the tumor cells
            TUintmax (torch.tensor): maximum number of interactions
            TUps (torch.tensor): probability of symmetric proliferation     

        Args:
            config (dict): initial state
            input_variables (dict): input variables
            output_variables (dict): output variables
            arguments (dict): arguments
            
        Returns:
            state (dict): updated state
    
    """

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state, action):
        location_matrix = get_by_path(state, re.split("/", self.input_variables["location_matrix"]))
        TUprolmax = get_by_path(state, re.split("/", self.input_variables["TUprolmax"]))
        Tum_proliferation_status = get_by_path(state, re.split("/", self.input_variables["Tum_proliferation_status"]))
        Tum_engagement_status = get_by_path(state, re.split("/", self.input_variables["Tum_engagement_status"]))
        TUintmax = get_by_path(state, re.split("/", self.input_variables["TUintmax"]))
        TUps = get_by_path(state, re.split("/", self.input_variables["TUps"]))
        prolif_action = action['tumorcells']['prolif_action']
        event_weight = action['tumorcells']['raw_weight']
        soft_tumor_delta = get_by_path(state, re.split("/", self.input_variables["soft_tumor_delta"]))
        TUMOR_PROLIFERATION_CAPACITY = 10
        TUMOR_INTERACTION_CAPACITY = 2

        num_agents, grid_height, grid_width = location_matrix.shape

        # work on clones and mutate directly, rather than rebuilding every
        # agent's tensor via a per-agent helper — we need cross-agent bookkeeping
        # (which dead slots are still free) that a purely per-agent loop can't do.
        updated_location_matrix = location_matrix.clone()
        updated_TUprolmax = TUprolmax.clone()
        updated_TUintmax = TUintmax.clone()
        updated_Tum_proliferation_status = Tum_proliferation_status.clone()
        updated_Tum_engagement_status = Tum_engagement_status.clone()

        # a "dead" slot = an agent index with no live cell. These are the
        # only valid homes for a newly proliferated daughter cell.
        dead_slot_mask = (location_matrix.sum(dim=(1, 2)) == 0)
        dead_slot_indices = torch.where(dead_slot_mask)[0].tolist()
        dead_slot_ptr = 0

        # cells already occupied by ANY live agent this step, so two
        # daughters (or a daughter and an unrelated cell) can't land on the same pixel.
        claimed_map = (location_matrix.sum(dim=0) > 0).float()

        # differentiable shadow accumulator — track which parents actually fired
        fired_mask = torch.zeros(num_agents)

        for agent_idx in range(num_agents):
            if dead_slot_mask[agent_idx]:
                continue  # no parent here, nothing to proliferate

            matrix = location_matrix[agent_idx]
            probs = prolif_action[agent_idx]

            old_y, old_x = torch.where(matrix == 1)
            if len(old_x) == 0:
                continue

            # only look for a DAUGHTER site distinct from the parent's own
            # cell, and exclude anything already claimed by another agent this step
            candidate_probs = probs.clone()
            candidate_probs[old_y, old_x] = 0
            candidate_probs = candidate_probs * (1 - claimed_map)

            cand_y, cand_x = torch.where(candidate_probs > 0.5)
            if cand_y.numel() == 0:
                continue  # this agent did not proliferate this step

            if dead_slot_ptr >= len(dead_slot_indices):
                continue  # no free agent slot left to host a daughter (at capacity)

            softmax_probs = F.softmax(candidate_probs[cand_y, cand_x], dim=0)
            pick = softmax_probs.multinomial(1, False)
            new_y, new_x = cand_y[pick], cand_x[pick]

            daughter_idx = dead_slot_indices[dead_slot_ptr]
            dead_slot_ptr += 1

            # daughter is written into its OWN agent slot — parent's slot
            # (agent_idx) is left completely untouched, preserving one-cell-per-agent.
            updated_location_matrix[daughter_idx] = torch.zeros(grid_height, grid_width)
            updated_location_matrix[daughter_idx][new_y, new_x] = 1
            updated_TUprolmax[daughter_idx] = torch.zeros(grid_height, grid_width)
            updated_TUprolmax[daughter_idx][new_y, new_x] = TUMOR_PROLIFERATION_CAPACITY
            updated_TUintmax[daughter_idx] = torch.zeros(grid_height, grid_width)
            updated_TUintmax[daughter_idx][new_y, new_x] = TUMOR_INTERACTION_CAPACITY
            updated_Tum_proliferation_status[daughter_idx] = torch.zeros(grid_height, grid_width)
            updated_Tum_proliferation_status[daughter_idx][new_y, new_x] = 1
            updated_Tum_engagement_status[daughter_idx] = torch.zeros(grid_height, grid_width)

            fired_mask[agent_idx] = 1.0
            claimed_map[new_y, new_x] = 1

        # differentiable delta — fired_mask is a constant (built from hard
        # torch.where/multinomial decisions, detached implicitly), event_weight
        # carries the theta-dependence via Gumbel-softmax. Gradient flows here.
        step_delta = (fired_mask.detach() * event_weight).sum()
        updated_soft_tumor_delta = soft_tumor_delta + step_delta

        # print("tumor cell proliferation transition complete! (%d new cells)" % dead_slot_ptr)
        return {self.output_variables[0]: updated_location_matrix,
                self.output_variables[1]: updated_TUprolmax,
                self.output_variables[2]: updated_Tum_engagement_status,
                self.output_variables[3]: updated_Tum_proliferation_status,
                self.output_variables[4]: updated_TUintmax,
                self.output_variables[5]: updated_soft_tumor_delta}
