import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import matplotlib.pyplot as plt
import ipdb

"""
    1. Updating the engagement state of tumor cells.
    2. Updating the state of immune cells.
"""

class Combat(SubstepTransition):
    """
        Transition function for the combat substep
        The neighborhood matrix consist of the tumor cells around of the immune cells
        The immune cells kill the tumor cells with the maximum killing capacity
        and the original location for the immune cells from the location matrix is provided
        Update the state of the grid consisting of both tumor and immune cells after
        the combat with change in the parameters of the cells

        Args:
            config (dict): the initial state
            input_variables (dict): dictionary of input variables
            output_variables (dict): dictionary of output variables
            arguments (dict): dictionary of arguments

        Returns:
            updated location matrix : the updated location matrix of the immune cells

        Parameters:

            IMkmax (tensor): the maximum interaction (killing) capacity of an immune cell, 
                the length to which the immune cell can survive
            kill_cell_index (tensor): the index of the tumor cell that is killed
            CD8_engagement_status (tensor): the engagement status of the immune cell
            TUintmax (tensor): the maximum interaction capacity of a tumor cell with immune cells, 
                length to which the tumor cell can survive
            Tum_engagement_status (tensor): the engagement status of the tumor cell
            engagementDuration (tensor): the engagement duration between the tumor and immune cells
            location matrix (tensor): the location matrix of the immune cells
            neighborhood matrix (tensor): the neighborhood matrix of the immune cells

        Steps:
            1. Feed the initial state to the transition function
            2. Get the input variables from the state and the target cells using action substep
            3. Rule is with each iteration of the engagementDuration parameter, the immune
            killing capacity (IMkmax parameter) gets reduced by one, while engagementDuration
            continuously drops until it becomes negative. EngagementDuration stays 1 during combat
            4. Each attack decreases the immune killing capacity (IMkmax) and tumor killing capacity
            (TUintmax) by 1, and the engagementDuration by 1.
            5. If the engagementDuration becomes negative, the combat ends and the cells survive.
            6. If the IMkmax or TUintmax becomes negative, that cell dies.
            7. Update the state of the grid with the new parameters of the cells
            
    """
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
    
    # kill_x, kill_y = kill cell index (tumor cell)
    # im_x, im_y = immune cell index
    def forward(self, state, action):
        selected_immune_cells = action['immunecells']['selected_immune_cells']
        tumor_location_matrix = get_by_path(state, re.split("/", self.input_variables["location_matrix"]))
        immune_location_matrix = get_by_path(state, re.split("/", self.input_variables["immune_location_matrix"]))
        engagementDuration = get_by_path(state, re.split("/", self.input_variables["engagementDuration"])) 
        # combined_cell_matrix = get_by_path(state, re.split("/", self.input_variables["combined_cell_matrix"]))
        IMkmax = get_by_path(state, re.split("/", self.input_variables["IMkmax"])) 
        TUintmax = get_by_path(state, re.split("/", self.input_variables["TUintmax"]))
        CD8_engagement_status = get_by_path(state, re.split("/", self.input_variables["CD8_engagement_status"]))
        Tum_engagement_status = get_by_path(state, re.split("/", self.input_variables["Tum_engagement_status"]))

        for agent_idx in range(len(selected_immune_cells)):
            # CHANGED: skip tumor cells with no eligible immune cell selected (empty list)
            if len(selected_immune_cells[agent_idx]) == 0:
                continue

            tumor_locations = torch.where(tumor_location_matrix[agent_idx] == 1)
            if len(tumor_locations[0]) == 0:
                continue
            kill_x, kill_y = tumor_locations[0][0].item(), tumor_locations[1][0].item()
            im_x, im_y = selected_immune_cells[agent_idx][0][1], selected_immune_cells[agent_idx][0][2] #[IM_agent, x, y]
            immune_cell_idx = selected_immune_cells[agent_idx][0][0]
            engagementDuration_value = engagementDuration[immune_cell_idx][im_x, im_y]
            IM_health = IMkmax[immune_cell_idx][im_x, im_y]
            TU_health = TUintmax[agent_idx][kill_x, kill_y]

            for step in range(int(engagementDuration_value)):
                if (IM_health <= 0) or (TU_health <= 0):
                    break
                else:
                    IM_health = IM_health - torch.tensor(1, dtype=torch.float32)
                    TU_health = TU_health - torch.tensor(1, dtype=torch.float32)

            IM_survived = (IM_health > 0)
            TU_survived = (TU_health > 0)
            immune_location_matrix[immune_cell_idx][im_x, im_y] *= IM_survived.int().item()
            tumor_location_matrix[agent_idx][kill_x, kill_y] *= TU_survived.int().item()

            IMkmax[immune_cell_idx][im_x, im_y] = IM_health
            TUintmax[agent_idx][kill_x, kill_y] = TU_health
            CD8_engagement_status[immune_cell_idx][im_x, im_y] = 0
            Tum_engagement_status[agent_idx][kill_x, kill_y] = 0

        # print('combat transition completed!')
        return {self.output_variables[0]: tumor_location_matrix,
                self.output_variables[1]: immune_location_matrix,
                self.output_variables[2]: IMkmax,
                self.output_variables[3]: TUintmax,
                self.output_variables[4]: CD8_engagement_status,
                self.output_variables[5]: Tum_engagement_status}
