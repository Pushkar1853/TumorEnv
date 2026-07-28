import torch
from AgentTorch.helpers import read_config
from AgentTorch import Runner, Registry

def create_registry():
    reg = Registry()

    return reg

config_path = ''

config = read_config(config_path)
runner = Runner(config)
runner.init() # initialize all state properties in the simulation

frozen_seg_nn = MySegNN()
calibration_nn = MyCalibNN()
optimizer = torch.optim.SGD(list(calibration_nn.parameters()))

MyLossFn = torch.nn.L2Loss()

# optimizer = torch.optim.SGD(list(calibration_nn.parameters() + runner.parameters()))

# tumor image -> SegNN -> segmentation features -> CalibNN -> simulation_parameters -> runner.step_from_params(num_steps, simulation_parameters)

for episode in range(num_episodes):
    runner.reset()
    optimizer.zero_grad()

    data_batch = get_biopsy_data()
    data_features = frozen_seg_nn(data_batch)
    simulation_params = calibration_nn(data_features)

    runner.step_from_params(num_steps_per_episode, simulation_params)

    output = compute_output(runner.trajectory) # generate histogram

    loss = MyLossFn(output, ground_truth) # eg: ground_truth is histogram of biopsy
    loss.backward()
    optimizer.step()