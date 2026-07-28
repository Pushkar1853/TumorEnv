import re
from functools import reduce
import operator
import torch
import copy
from omegaconf import OmegaConf
import pandas as pd

def get_by_path(root, items):
    r"""
        Access a nested object in root by item sequence
    """
    return reduce(operator.getitem, items, root)

def set_by_path(root, items, value):
    r"""
        Set a value in a nested object in root by item sequence
    """
    val_obj = get_by_path(root, items[:-1])
    val_obj[items[-1]] = value
    return root

def del_by_path(root, items):
    """Delete a key-value in a nested object in root by item sequence."""
    del get_by_path(root, items[:-1])[items[-1]]

def copy_module(dict_to_copy):
    r"""
        Creates a new dictionary with a copy of each PyTorch tensor in the input dictionary.
        Handles nested dictionaries of PyTorch tensors of variable depth.
    """
    copied_dict = {}
    for key, value in dict_to_copy.items():
        if torch.is_tensor(value):
            copied_dict[key] = torch.clone(value)
        elif isinstance(value, dict):
            copied_dict[key] = copy_module(value)
        elif not torch.is_tensor(value):
            copied_dict[key] = copy.deepcopy(value)
        else:
            raise TypeError("Type error.. ", type(value))
            
    return copied_dict

def process_shape(config, s):
    if type(s) == str:
        return get_by_path(config, re.split('/', s))
    else:
        return s

def read_config(config_file):
    # register OmegaConf resolvers for composite questions in OmegaConf
    try:
        OmegaConf.register_new_resolver("sum", lambda x, y: x + y)
        OmegaConf.register_new_resolver("multiply", lambda x, y: x*y)
    except:
        print("resolvers already registered..")
    
    if config_file[-5:] != ".yaml":
        raise ValueError("Config file type should be yaml")
    try:
        config = OmegaConf.load(config_file)
        config = OmegaConf.to_object(config)
    except Exception as e:
        raise ValueError(f"Could not load config file. Please check path ad file type. Error message is {str(e)}")

    return config

def read_from_file(shape, params):    
    file_path = params['file_path']
    
    if file_path[-3:] == 'csv':
        data = pd.read_csv(file_path)
    
    data_values = data.values
    assert data_values.shape == tuple(shape)
    
    data_tensor = torch.from_numpy(data_values)
        
    return data_tensor

def initialize_location_matrix(shape, params):
    file_path = params['file_path']

    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use a CSV file.")
    
    data_values = data.values
    grid_height, grid_width = shape
    data_values_tensor = torch.tensor(data_values, dtype=torch.float32)
    var = torch.reshape(data_values_tensor, (grid_height, grid_width))
    
    # Count the number of agents (cells with value 1 or 2)
    num_agents = int((var == 1).sum() + (var == 2).sum())
    
    # Initialize the location matrix with the correct shape
    location_matrix = torch.zeros((num_agents, grid_height, grid_width))
    
    # Create masks for tumor cells (1) and immune cells (2)
    tumor_mask = (var == 1)
    immune_mask = (var == 2)
    
    # Get the indices of tumor and immune cells
    tumor_indices = torch.nonzero(tumor_mask)
    immune_indices = torch.nonzero(immune_mask)
    
    # Assign values to the location matrix
    for idx, (i, j) in enumerate(tumor_indices):
        location_matrix[idx, i, j] = 1
    
    for idx, (i, j) in enumerate(immune_indices, start=len(tumor_indices)):
        location_matrix[idx, i, j] = 2
    
    # print(f"Initialized location matrix with shape: {location_matrix.shape}")
    return location_matrix

def initialize_parameter(shape, params):
    file_path = params['file_path']

    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    
    data_values = data.values
    values = params['values']
    num_agents = int(data_values.sum())
    grid_height, grid_width = shape
    data_values_tensor = torch.tensor(data_values, dtype=torch.float32)
    var = torch.reshape(data_values_tensor, (grid_height, grid_width))
    
    # Initialize with zeros
    parameter_matrix = torch.zeros((num_agents, grid_height, grid_width, 1), dtype=torch.float32)
    
    agent_idx = 0
    for i in range(grid_height):
        for j in range(grid_width):
            if var[i][j] >= 1:
                # Set the agent's position to value from the values
                parameter_matrix[agent_idx][i][j][0] = values
                agent_idx += 1
    
    return parameter_matrix

def reinitialize_location_matrix(num_agents, grid_height, grid_width, location_matrix):
    aggregated_image_new = torch.zeros((grid_height, grid_width))
    for agent_idx in range(num_agents):
        aggregated_image_new += location_matrix[agent_idx]
    new_num_agents = int(aggregated_image_new.sum())
    # reintialize the location matrix
    updated_location_matrix = torch.zeros((new_num_agents, grid_height, grid_width))
    agent_idx = 0
    for i in range(grid_height):
        for j in range(grid_width):
            if aggregated_image_new[i][j] == 1:
                updated_location_matrix[agent_idx][i][j] = 1
                agent_idx += 1

    return updated_location_matrix

def reinitialize_parameter_matrix(num_agents, grid_height, grid_width, parameter_matrix):
    aggregated_image_new = torch.zeros((grid_height, grid_width))
    for agent_idx in range(num_agents):
        aggregated_image_new += parameter_matrix[agent_idx, :, :]
    new_num_agents = int(aggregated_image_new.sum())
    # reintialize the location matrix
    updated_parameter_matrix = torch.zeros((new_num_agents, grid_height, grid_width, 1))
    agent_idx = 0
    # the parameter matrix is of shape (num_agents, grid_height, grid_width, 1)
    # but the num_agents of parameter matrix is less than the num_agents of updated_parameter_matrix
    # this parameter has to be assigned to the updated_parameter_matrix wherever it is one with the same agent_idx
    # the value will be the same as the previous value
    for i in range(grid_height):
        for j in range(grid_width):
            if aggregated_image_new[i][j] == 1:
                updated_parameter_matrix[agent_idx, i, j] = parameter_matrix[agent_idx, i, j]
                agent_idx += 1
    
    return updated_parameter_matrix

def initialize_combined_cell_matrix(shape, params):
    tumor_file_path = params['tumor_file_path']
    immune_file_path = params['immune_file_path']

    tumor_data = pd.read_csv(tumor_file_path).values
    immune_data = pd.read_csv(immune_file_path).values

    grid_height, grid_width = shape

    # Ensure the data matches the expected shape
    if tumor_data.shape != (grid_height, grid_width):
        tumor_data = tumor_data.reshape(grid_height, grid_width)
    if immune_data.shape != (grid_height, grid_width):
        immune_data = immune_data.reshape(grid_height, grid_width)

    combined_matrix = torch.zeros((grid_height, grid_width), dtype=torch.int)
    
    combined_matrix[torch.from_numpy(tumor_data) > 0] = 1
    combined_matrix[torch.from_numpy(immune_data) > 0] = 2
    
    return combined_matrix