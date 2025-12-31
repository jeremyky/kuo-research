import argparse
import json
import os
import random
from metadrive.engine.engine_utils import get_global_config, get_engine
from metadrive.engine.asset_loader import AssetLoader
from partial_manual_policy import HybridEgoCarPolicy, HybridEgoCarReplayPolicy
from metadrive.envs.custom_scenario_env import CustomScenarioEnv
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.scenario.utils import read_dataset_summary, read_scenario_data
from gymnasium.spaces import Discrete
from PIL import Image


os.environ["SDL_VIDEODRIVER"] = "dummy"  # Hide the pygame window
waymo_data = AssetLoader.file_path(AssetLoader.asset_path, "waymo", unix_style=False)
scenario_summary = read_dataset_summary(waymo_data)
for file in scenario_summary[1]:
    print(file)
    name = file.split('_')[2].split('.')[0]
    print(name)
    config = {
        "instructions": [
            "turn left at the intersection"
        ],
        "onset_manuals": [
            8
        ],
        "onset_manual": 8,
        "instructions_legacy": "turn left at the intersection",
        "reactive_traffic": True,
        "extra_steps": 10
        }
    with open(f'configs/{name}', 'w') as f:
        json.dump(config, f)