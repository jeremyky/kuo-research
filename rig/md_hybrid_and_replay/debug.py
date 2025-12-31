from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
import os
from metadrive.engine.asset_loader import AssetLoader
import metadrive.scenario.utils as sd_utils
from metadrive.scenario.scenario_description import ScenarioDescription as SD
import numpy as np

waymo_data = AssetLoader.file_path(AssetLoader.asset_path, "waymo", unix_style=False)

def make_line(x_offset, height, y_dir=1, color=(1,105/255,180/255)):
    points = [(x_offset+x,x*y_dir,height*x/10+height) for x in range(10)]
    colors = [np.clip(np.array([*color,1])*(i+1)/11, 0., 1.0) for i in range(10)]
    if y_dir<0:
        points = points[::-1]
        colors = colors[::-1]
    return points, colors

def make_lines_from_xy(x_start, y_start, height=1.0, length=10, y_dir=1.0, color=(1, 105/255, 180/255)):
    points = [(x_start + i, y_start + i * y_dir, height + (height * i / 10)) for i in range(length)]
    colors = [np.clip(np.array([*color, 1.0]) * (i + 1) / (length + 1), 0.0, 1.0) for i in range(length)]
    return points, colors




# create environment
env = ScenarioEnv(dict(use_render=True, 
                        show_coordinates=True, 
                        agent_policy=ReplayEgoCarPolicy,
                        num_scenarios=1,
                        start_scenario_index=1,
                        data_directory= waymo_data))

summary = sd_utils.read_dataset_summary(waymo_data)
scenario = sd_utils.read_scenario_data(os.path.join(waymo_data, summary[2][summary[1][0]], summary[1][0]))

env.reset() # launch the simulation
try:
    drawer = env.engine.make_line_drawer(thickness=5)
    for i in range(90):
        o, r, tm, tc, info = env.step([0, 0])
        if tm or tc:
            break

finally:    
    env.close()