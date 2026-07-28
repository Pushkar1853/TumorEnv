import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import get_by_path
import ipdb

"""
    1. Selecting which immune cells will attempt to kill.
    2. Choosing a target tumor cell for each active immune cell.
"""

class CombatDecision(SubstepAction):
    """
        Action function for the combat substep
        This function is used to select the immune cells that will attempt to kill
        and the target tumor cell for each active immune cell

        Args:
            config (dict): the initial state
            input_variables (dict): dictionary of input variables
            output_variables (dict): dictionary of output variables
            arguments (dict): dictionary of arguments

        Returns:
            immune_cell_id_list (tensor): the immune cell id list
            tumor_cell_id_list (tensor): the tumor cell id list

        Parameters:
            eligible immune cells (tensor): the eligible immune cells
            IMpkill (float): killing probability of immune cells
            neighborhood tumor matrices (tensor): the neighborhood tumor matrices

        Steps:
            1. Feed the initial state to the action function
            2. Read the parameters from state as variables from the action substep
            3. Select the immune cell randomly
            4. Select the target tumor cell randomly
            5. Return the immune cell id list and the tumor cell id list

    """
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        # self.temperature = 1.0  # Adjustable temperature for Gumbel-Softmax
        self.IMpkill = torch.nn.Parameter(torch.tensor(0.6)) 

    def forward(self, state, observation):
        eligible_immune_cells = observation['eligible_immune_cells']
        IMpkill = get_by_path(state, re.split("/", self.input_variables["IMpkill"]))  # CHANGED
        selected_immune_cells = {}

        for tumor_idx in eligible_immune_cells:
            tumor_eligible_cells = torch.tensor(eligible_immune_cells[tumor_idx])

            if len(tumor_eligible_cells) == 0:
                selected_immune_cells[tumor_idx] = []
                continue

            # CHANGED: gate combat on IMpkill — even if an immune cell is adjacent,
            # it only actually attacks with probability IMpkill this step
            attack_roll = torch.rand(1).item()
            if attack_roll > IMpkill.item():
                selected_immune_cells[tumor_idx] = []
                continue

            probs = torch.ones(len(tumor_eligible_cells))
            softmax_probs = F.softmax(probs, dim=0)
            selected_idx = softmax_probs.multinomial(1, False)
            selected_immune_cells[tumor_idx] = tumor_eligible_cells[selected_idx].tolist()

        # print('combat policy completed!')
        return {self.output_variables[0]: selected_immune_cells}
