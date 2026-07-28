import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import get_by_path

class MigrationTUDecision(SubstepAction):
    """
        Action function for the migration substep
        Decide whether the tumor cell migrates or not and send the
        migrate cell index to the transition function

        Steps:
            1. The action function receives input as initial and new neighborhood of the tumor cell
            2. The function follows a rule that if the probability of that agent cell is less 
                than the migration probability, then the agent cell will migrate to the new location.
            3. The following function is made differentiable by using the Gumbel softmax function
            4. If the probability is greater than the migration probability, then the agent cell will
                remain at the same location, meaning the location matrix gets passed and the 
                agent cell will remain at the same location.
            5. We use the Gumbel softmax function to remove the hard classifier
            6. This passes the obtained neighborhood matrix to the transition function

        Parameters:
            TUpmig (tensor): the probability of the tumor cell migrating
            TUdeath (tensor): the probability of the tumor cell dying
            TUpprol (tensor): the probability of the tumor cell proliferating
            initial_neighborhood (tensor): the initial neighborhood of the tumor cell
            new_neighborhood (tensor): the new neighborhood of the tumor cell
            
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
        self.sigmoid_scale = 10

    def _get_nature(self, TUpmig, TUpdeath, TUpprol):
        return torch.stack([TUpprol, TUpdeath, TUpmig])

    def forward(self, state, observation, tau=0.5):
        TUpmig = get_by_path(state, re.split("/", self.input_variables["TUpmig"]))
        TUpprol = get_by_path(state, re.split("/", self.input_variables["TUpprol"]))
        TUpdeath = get_by_path(state, re.split("/", self.input_variables["TUpdeath"]))

        new_neighborhood = observation["new_neighborhood"]
        initial_neighborhood = observation["location_matrix"]

        num_agents = initial_neighborhood.shape[0]
        logits = self._get_nature(TUpmig, TUpdeath, TUpprol).view(3, 1).expand(3, num_agents)

        gumbel_softmax_sample = F.gumbel_softmax(logits, tau=tau, dim=0, hard=True)  # [3, num_agents]
        new_weight = 0.7 * gumbel_softmax_sample[0]           # [num_agents]
        new_weight = new_weight.view(-1, 1, 1)                # broadcast over H, W

        migrate_action = (1 - new_weight) * initial_neighborhood + new_weight * new_neighborhood

        # print("tumor migration decision taken with probabilities!")
        return {self.output_variables[0]: migrate_action}
