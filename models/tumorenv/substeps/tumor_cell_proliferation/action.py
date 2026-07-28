import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import get_by_path

class ProliferationDecision(SubstepAction):
    """
        Action function for the proliferation substep
        Decide whether the tumor cell proliferates or not and send the
        proliferate cell index to the transition function

        Steps:
            1. The action function receives input as initial and new neighborhood of the tumor cell
            2. The function follows a rule that if the probability of that agent cell is less 
                than the proliferation probability, then the agent cell will migrate to the new location.
            3. The following function is made differentiable by using the Gumbel softmax function
            4. If the probability is greater than the proliferation probability, then the agent cell will
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
        initial_neighborhood = observation["tumor_location_matrix"]

        # CHANGED: broadcast the 3 global probabilities into per-agent logits so
        # every tumor cell draws its OWN independent Gumbel-softmax sample this step,
        # instead of the whole population sharing a single coin flip (which is why
        # proliferation was either "everyone" or "no one" from step to step).
        num_agents = initial_neighborhood.shape[0]
        logits = self._get_nature(TUpmig, TUpdeath, TUpprol).view(3, 1).expand(3, num_agents)

        gumbel_softmax_sample = F.gumbel_softmax(logits, tau=tau, dim=0, hard=True)  # [3, num_agents]
        new_weight = 0.7 * gumbel_softmax_sample[0]           # [num_agents]
        new_weight = new_weight.view(-1, 1, 1)                # broadcast over H, W

        prolif_action = (1 - new_weight) * initial_neighborhood + new_weight * new_neighborhood

        # print("tumor proliferation decision taken with probabilities!")
        return {self.output_variables[0]: prolif_action}