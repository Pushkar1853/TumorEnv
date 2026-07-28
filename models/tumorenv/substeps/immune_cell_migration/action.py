import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import get_by_path


class MigrationIMDecision(SubstepAction):
    """
        Action function for the migration substep
        Decide whether the immune cell migrates or not and send the
        neighborhood matrix to the transition function
        or the location matrix if not proliferating 

        Steps:
            1. The action function recieves input as initial and new neighborhood of the immune cell
            2. The function follows a rule that if the probability of that agent cell is less 
                than the migration probability, then the agent cell will migrate to the new location.
            3. The following function is made differentiable by using the Gumbel softmax function
            4. If the probability is greater than the migration probability, then the agent cell will
                remain at the same location, meaning the location matrix gets passed and the 
                agent cell will remain at the same location.
            5. We use the Gumbel softmax function to remove the hard classifier
            6. This passes the obtained neighborhood matrix to the transition function

        Parameters:
            IMpmig (tensor): the probability of the immune cell migrating
            IMdeath (tensor): the probability of the immune cell dying
            IMpprol (tensor): the probability of the immune cell proliferating
            new_neighborhood (tensor) : the neighborhood matrix of the immune cells consisting of empty locations
            initial_neighborhood (tensor) : the location matrix of the immune cells (consisting original locations)

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

    def _get_nature(self, IMpmig, IMpdeath, IMpprol):
        return torch.stack([IMpprol, IMpdeath, IMpmig])
    
    def forward(self, state, observation, tau=0.5):
        input_variables = self.input_variables
        IMpmig = get_by_path(state, re.split("/", input_variables["IMpmig"]))
        IMpdeath = get_by_path(state, re.split("/", input_variables["IMpdeath"]))
        IMpprol = get_by_path(state, re.split("/", input_variables["IMpprol"]))

        new_neighborhood = observation["new_neighborhood"]
        initial_neighborhood = observation["immune_location_matrix"]

        num_agents = initial_neighborhood.shape[0]
        logits = self._get_nature(IMpmig, IMpdeath, IMpprol).view(3, 1).expand(3, num_agents)

        gumbel_softmax_sample = F.gumbel_softmax(logits, tau=tau, dim=0, hard=True)  # [3, num_agents]
        new_weight = 0.7 * gumbel_softmax_sample[0]           # [num_agents]
        new_weight = new_weight.view(-1, 1, 1)                # broadcast over H, W

        migrate_action = (1 - new_weight) * initial_neighborhood + new_weight * new_neighborhood
        
        # print("immune cell migration action complete!")
        return {self.output_variables[0]: migrate_action}