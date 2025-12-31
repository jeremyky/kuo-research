import os
import json
from metadrive.envs.scenario_env import ScenarioEnv
import cv2
import numpy as np
from metadrive.component.sensors.rgb_camera import RGBCamera

def load_mapping():
    """Load the mapping file to find our converted scenarios."""
    mapping_file = os.path.abspath("data/waymo/waymo_data_mapping.json")
    with open(mapping_file, 'r') as f:
        return json.load(f)

def main():
    # Create output directory for frames
    output_dir = "scenario_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load our data mapping
    mapping = load_mapping()
    metadrive_dir = mapping["metadrive_scenarios"]["directory"]
    
    # Configure MetaDrive environment with visualization
    config = {
        "use_render": False,  # Disable onscreen rendering
        "image_observation": True,  # Enable image observations
        "debug": True,  # Enable debug mode to see more information
        "manual_control": False,  # Disable manual control since we can't use keyboard
        
        # Vehicle configuration
        "vehicle_config": {
            "show_navi_mark": False,
            "show_dest_mark": False,
            "show_line_to_dest": False,
            "image_source": "rgb_camera",  # Use RGB camera for observations
        },
        
        # Sensor configuration
        "sensors": {
            "rgb_camera": (RGBCamera, 800, 800),  # Add RGB camera sensor
        },
        
        # Environment configuration
        "data_directory": metadrive_dir,
        "num_scenarios": 1,
        "start_scenario_index": 0,
        
        # Rendering configuration
        "window_size": (800, 800),
        "show_interface": False,
        "show_fps": False,
        "force_render_fps": 20,
        "render_pipeline": False,
        
        # Disable unnecessary features
        "show_coordinates": False,
        "show_skybox": False,
        "show_terrain": False,
        "show_logo": False,
        "show_mouse": False,
    }
    
    # Create and initialize the environment
    env = ScenarioEnv(config)
    
    print("\nInitializing MetaDrive environment...")
    print(f"Loading scenarios from: {metadrive_dir}")
    print(f"Saving frames to: {output_dir}")
    
    # Reset to start the scenario
    obs, info = env.reset()
    
    print("\nScenario loaded!")
    print("\nRunning simulation and saving frames...")
    
    # Main simulation loop
    done = False
    step = 0
    total_reward = 0
    try:
        while not done and step < 100:  # Limit to 100 steps for testing
            # Take a random action (you can modify this)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            # Save the observation as an image
            if isinstance(obs, dict) and "image" in obs:
                frame = obs["image"]
            else:
                frame = obs
            
            if frame is not None:
                # Convert to uint8 if needed
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Save the frame
                frame_path = os.path.join(output_dir, f"frame_{step:04d}.png")
                cv2.imwrite(frame_path, frame)
                print(f"Saved frame {step} to {frame_path}")
            
            if done:
                print(f"\nScenario complete after {step} steps!")
                print(f"Final total reward: {total_reward:.2f}")
                print(f"Termination reason: {'terminated' if terminated else 'truncated'}")
                if info:
                    print("Additional info:", info)
                break
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()  # Print full error traceback
    finally:
        env.close()
    
    print(f"\nSimulation complete! {step} frames saved to {output_dir}")
    print("You can view the frames using an image viewer or create a video from them.")

if __name__ == "__main__":
    main() 