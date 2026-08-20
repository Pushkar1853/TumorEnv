from AgentTorch import Runner, Registry
import torch
import gc

def TUIM_registry():
    reg = Registry()

    # Transition
    from substeps.tumor_cell_migration.transition import MigrateTUCell
    reg.register(MigrateTUCell, "MigrateTUCell", key="transition")
    from substeps.immune_cell_migration.transition import MigrateIMCell
    reg.register(MigrateIMCell, "MigrateIMCell", key="transition")
    from substeps.immune_cell_influx.transition import InfluxIMCell
    reg.register(InfluxIMCell, "InfluxIMCell", key="transition")
    from substeps.immune_cell_proliferation.transition import ProliferateIMCell
    reg.register(ProliferateIMCell, "ProliferateIMCell", key="transition")
    from substeps.tumor_cell_proliferation.transition import ProliferateTUCell
    reg.register(ProliferateTUCell, "ProliferateTUCell", key="transition")
    from substeps.tumor_cell_death.transition import KillTUCell
    reg.register(KillTUCell, "KillTUCell", key="transition")
    from substeps.immune_cell_death.transition import KillIMCell
    reg.register(KillIMCell, "KillIMCell", key="transition")
    from substeps.tumor_immune_cell_interaction.transition import Combat
    reg.register(Combat, "Combat", key="transition")

    # Action
    from substeps.tumor_cell_migration.action import MigrationTUDecision
    reg.register(MigrationTUDecision, "MigrationTUDecision", key="policy")
    from substeps.immune_cell_migration.action import MigrationIMDecision
    reg.register(MigrationIMDecision, "MigrationIMDecision", key="policy")
    from substeps.tumor_cell_proliferation.action import ProliferationDecision
    reg.register(ProliferationDecision, "ProliferationDecision", key="policy")
    from substeps.immune_cell_proliferation.action import ProliferationIMDecision
    reg.register(ProliferationIMDecision, "ProliferationIMDecision", key="policy")
    from substeps.immune_cell_influx.action import InfluxIMDecision
    reg.register(InfluxIMDecision, "InfluxIMDecision", key="policy")
    from substeps.tumor_immune_cell_interaction.action import CombatDecision
    reg.register(CombatDecision, "CombatDecision", key="policy")

    # Observation
    from substeps.tumor_cell_migration.observation import ObserveFarNeighborhood, GetFromState
    reg.register(GetFromState, "get_from_state", key="observation")
    reg.register(ObserveFarNeighborhood, "observe_far_neighborhood", key="observation")
    from substeps.tumor_cell_proliferation.observation import ObserveNeighborhood, GetFromState
    reg.register(GetFromState, "get_from_state", key="observation")
    reg.register(ObserveNeighborhood, "observe_neighborhood", key="observation")
    from substeps.immune_cell_migration.observation import  GetFromState, IM_ObserveFarNeighborhood
    reg.register(GetFromState, "get_from_state", key="observation")
    reg.register(IM_ObserveFarNeighborhood, "IM_ObserveFarNeighborhood", key="observation")
    from substeps.immune_cell_proliferation.observation import ObserveIMpNeighborhood, GetFromState
    reg.register(GetFromState, "get_from_state", key="observation")
    reg.register(ObserveIMpNeighborhood, "ObserveIMpNeighborhood", key="observation")
    from substeps.immune_cell_influx.observation import ObserveInfluxCandidates, GetFromState
    reg.register(GetFromState, "get_from_state", key="observation")
    reg.register(ObserveInfluxCandidates, "ObserveInfluxCandidates", key="observation")
    from substeps.tumor_immune_cell_interaction.observation import ObserveIMkNeighborhood, GetFromState
    reg.register(GetFromState, "get_from_state", key="observation")
    reg.register(ObserveIMkNeighborhood, "ObserveIMkNeighborhood", key="observation")

    from AgentTorch.helpers.environment import grid_network
    reg.register(grid_network, "grid", key="network")

    from AgentTorch.helpers.general import reinitialize_location_matrix, reinitialize_parameter_matrix
    reg.register(reinitialize_location_matrix, "reinitialize_location_matrix", key="initialization")
    reg.register(reinitialize_parameter_matrix, "reinitalize_parameter_matrix", key="initialization")

    from AgentTorch.helpers import read_from_file, initialize_location_matrix, initialize_parameter, initialize_combined_cell_matrix
    reg.register(read_from_file, 'read_from_file', key='initialization')
    reg.register(initialize_location_matrix, 'initialize_location_matrix', key='initialization')
    reg.register(initialize_parameter, 'initialize_parameter', key='initialization')
    reg.register(initialize_combined_cell_matrix, 'initialize_combined_cell_matrix', key='initialization')

    return reg
    
class TU_IM_Runner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_state = None
        self.final_state = None
        self.grid_trajectory = []  # CHANGED: lightweight replacement for state_trajectory
        self._initial_fractions = (0.0, 0.0)

    @staticmethod
    def _state_to_grid(state, device='cpu'):
        tl = state['agents']['tumorcells']['TU_location_matrix']
        il = state['agents']['immunecells']['IM_location_matrix']
        t2d = torch.sum(tl, dim=0)
        i2d = torch.sum(il, dim=0)
        g = torch.zeros_like(t2d, device=device, dtype=torch.int)
        g[t2d > 0] = 1
        g[i2d > 0] = 2
        return g.detach().cpu()

    def step(self, num_steps=None):
        # CHANGED: override the library's default step() — it stores every full
        # per-agent [N,H,W] state tensor (~4GB) twice per substep. We only need
        # the collapsed 2D grid for visualization, so convert and discard immediately.
        assert self.state is not None
        if not num_steps:
            num_steps = self.config["simulation_metadata"]["num_steps_per_episode"]

        device = self.config['simulation_metadata'].get('device', 'cpu')
        self.grid_trajectory.append([self._state_to_grid(self.state, device)])

        for time_step in range(num_steps):
            self.state['current_step'] = time_step

            for substep in self.config['substeps'].keys():
                observation_profile, action_profile = {}, {}
                for agent_type in self.config['substeps'][substep]['active_agents']:
                    assert substep == self.state['current_substep']
                    assert time_step == self.state['current_step']
                    observation_profile[agent_type] = self.controller.observe(
                        self.state, self.initializer.observation_function, agent_type)
                    action_profile[agent_type] = self.controller.act(
                        self.state, observation_profile[agent_type], self.initializer.policy_function, agent_type)

                next_state = self.controller.progress(self.state, action_profile, self.initializer.transition_function)
                self.state = next_state
                self.grid_trajectory[-1].append(self._state_to_grid(self.state, device))

                del observation_profile, action_profile
                import gc; gc.collect()

    def forward(self):
        for episode in range(self.config['simulation_metadata']['num_episodes']):
            num_steps_per_episode = self.config["simulation_metadata"]["num_steps_per_episode"]
            self.reset()
            self.initial_state = self.get_current_state()
            self.step(num_steps_per_episode)
            self.final_state = self.get_current_state()
            torch.cuda.empty_cache()
            gc.collect()

    def execute(self):
        self.forward()

    def get_soft_population_fractions(self):
        from AgentTorch.helpers import get_by_path
        soft_tumor_delta = get_by_path(self.state, ["environment", "soft_tumor_delta"])
        soft_immune_delta = get_by_path(self.state, ["environment", "soft_immune_delta"])
        total_cells = self.config['simulation_metadata']['N'] * self.config['simulation_metadata']['M']
        initial_t, initial_i = self._initial_fractions
        return (initial_t + soft_tumor_delta / total_cells,
                initial_i + soft_immune_delta / total_cells)

    def reset(self):
        self.init()
        grid = self.get_current_state()
        t = (grid == 1).float().mean().detach()
        i = (grid == 2).float().mean().detach()
        self._initial_fractions = (t, i)

    def get_current_state(self):
        return self._state_to_grid(self.state)

    def get_initial_state(self):
        return self.initial_state

    def get_final_state(self):
        return self.final_state
