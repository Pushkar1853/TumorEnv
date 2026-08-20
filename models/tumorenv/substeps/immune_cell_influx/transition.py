import re
import torch
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path


class InfluxIMCell(SubstepTransition):
    """
        Activate freshly-influxed immune cells at their chosen grid location
        and reset their capacity parameters to full health. Agent slots not
        selected for influx this step pass through unchanged.
    """
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

    def forward(self, state, action):
        input_variables = self.input_variables

        immune_location_matrix = get_by_path(state, re.split("/", input_variables["immune_location_matrix"]))
        IMprolmax = get_by_path(state, re.split("/", input_variables["IMprolmax"]))
        IMkmax = get_by_path(state, re.split("/", input_variables["IMkmax"]))
        IMintmax = get_by_path(state, re.split("/", input_variables["IMintmax"]))
        engagementDuration = get_by_path(state, re.split("/", input_variables["engagementDuration"]))
        CD8_proliferation_status = get_by_path(state, re.split("/", input_variables["CD8_proliferation_status"]))
        CD8_non_proliferation_status = get_by_path(state, re.split("/", input_variables["CD8_non_proliferation_status"]))
        CD8_engagement_status = get_by_path(state, re.split("/", input_variables["CD8_engagement_status"]))
        soft_immune_delta = get_by_path(state, re.split("/", input_variables["soft_immune_delta"]))
        IMinfluxProb = get_by_path(state, re.split("/", input_variables["IMinfluxProb"]))

        influx_targets = action['immunecells']['influx_targets']

        IMMUNE_PROLIFERATION_CAPACITY = 10
        IMMUNE_INTERACTION_CAPACITY = 40
        IMMUNE_KILLING_CAPACITY = 5
        ENGAGEMENT_DURATION = 48

        newly_influxed = (influx_targets.sum(dim=(1, 2)) > 0).view(-1, 1, 1)

        updated_location_matrix = torch.where(newly_influxed, influx_targets, immune_location_matrix)
        updated_IMprolmax = torch.where(newly_influxed, influx_targets * IMMUNE_PROLIFERATION_CAPACITY, IMprolmax)
        updated_IMkmax = torch.where(newly_influxed, influx_targets * IMMUNE_KILLING_CAPACITY, IMkmax)
        updated_IMintmax = torch.where(newly_influxed, influx_targets * IMMUNE_INTERACTION_CAPACITY, IMintmax)
        updated_engagementDuration = torch.where(newly_influxed, influx_targets * ENGAGEMENT_DURATION, engagementDuration)
        updated_CD8_proliferation_status = torch.where(newly_influxed, torch.zeros_like(CD8_proliferation_status), CD8_proliferation_status)
        updated_CD8_non_proliferation_status = torch.where(newly_influxed, influx_targets, CD8_non_proliferation_status)
        updated_CD8_engagement_status = torch.where(newly_influxed, torch.zeros_like(CD8_engagement_status), CD8_engagement_status)

        # closed-form expected contribution — differentiable in IMinfluxProb directly
        # dead_mask computed from state: agent slots with no live cell
        dead_mask = (immune_location_matrix.sum(dim=(1, 2)) == 0)
        num_dead_slots = dead_mask.sum().float()
        step_delta = num_dead_slots.detach() * IMinfluxProb
        updated_soft_immune_delta = soft_immune_delta + step_delta

        return {self.output_variables[0]: updated_location_matrix,
                self.output_variables[1]: updated_IMprolmax,
                self.output_variables[2]: updated_IMkmax,
                self.output_variables[3]: updated_IMintmax,
                self.output_variables[4]: updated_engagementDuration,
                self.output_variables[5]: updated_CD8_proliferation_status,
                self.output_variables[6]: updated_CD8_non_proliferation_status,
                self.output_variables[7]: updated_CD8_engagement_status,
                self.output_variables[8]: updated_soft_immune_delta}
