

from IPython.display import Image as IImage
import pygame
import numpy as np
from PIL import Image
from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.envs.scenario_env import ScenarioEnv
import os
from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario import utils as sd_utils
waymo_data = "/Users/ningzeqiang/Desktop/Proejct/msdn/scenarionet/scenarionet/waymo_data" 
os.listdir(waymo_data) 
dataset_path = waymo_data
print("Dataset path: ", dataset_path)

# Get the scenario .pkl file name
_, scenario_ids, dataset_mapping = sd_utils.read_dataset_summary(dataset_path)
# Just pick the first scenario
scenario_pkl_file = scenario_ids[0]

# Get the relative path to the .pkl file
print("The pkl file relative path: ", dataset_mapping[scenario_pkl_file])  # An empty path

# Get the absolute path to the .pkl file
abs_path_to_pkl_file = os.path.join(dataset_path, dataset_mapping[scenario_pkl_file], scenario_pkl_file)
print("The pkl file absolute path: ", abs_path_to_pkl_file)

# Call utility function in MD and get the Scenario Description object
scenario = sd_utils.read_scenario_data(abs_path_to_pkl_file)

print(f"\nThe raw data type after reading the .pkl file is {type(scenario)}")
print(f"The keys in a ScenarioDescription are: {scenario.keys()}")
sdc_id = scenario['metadata'].keys()
print(f"The SDC vehicle type is: {scenario['tracks'][sdc_id]['type']}")
print(scenario['tracks'].keys())
print(scenario['tracks'][sdc_id].keys())
print("The state of object is", scenario['tracks'][sdc_id]['state'].keys())
print("The state of object is", scenario['tracks']["799"]['state']['position'])
print("The state of object is", len(scenario['tracks']["822"]['state']['velocity']))
