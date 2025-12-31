import matplotlib.pyplot as plt
import numpy as np
from reader import WaymoScenarioReader
import os
from matplotlib import animation
from matplotlib.patches import Rectangle
from typing import Dict, List

class WaymoScenarioVisualizer:
    def __init__(self, scenario_data: Dict):
        """Initialize the visualizer with parsed scenario data."""
        self.scenario_data = scenario_data
        self.fig, self.ax = None, None
        
        # Color mapping for different object types
        self.type_colors = {
            1: 'blue',      # TYPE_VEHICLE
            2: 'green',     # TYPE_PEDESTRIAN
            3: 'red',       # TYPE_CYCLIST
            4: 'gray'       # TYPE_OTHER
        }
    
    def setup_plot(self):
        """Set up the matplotlib figure and axes."""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        plt.title(f"Scenario {self.scenario_data['scenario_id']}")
    
    def plot_frame(self, frame_idx: int):
        """Plot a single frame of the scenario."""
        self.ax.clear()
        self.ax.grid(True)
        
        # Plot each track at this timestamp
        for track in self.scenario_data['tracks']:
            if frame_idx < len(track['states']):
                state = track['states'][frame_idx]
                if state['valid']:
                    # Plot position
                    color = self.type_colors.get(track['object_type'], 'gray')
                    marker = 'o' if track['id'] != self.scenario_data['sdc_track_index'] else 's'
                    size = 100 if track['id'] != self.scenario_data['sdc_track_index'] else 150
                    
                    self.ax.scatter(state['center_x'], state['center_y'], 
                                  c=color, marker=marker, s=size)
                    
                    # Plot velocity vector
                    if state['velocity_x'] != 0 or state['velocity_y'] != 0:
                        vel_scale = 1.0  # Scale factor for velocity vectors
                        self.ax.arrow(state['center_x'], state['center_y'],
                                    state['velocity_x'] * vel_scale,
                                    state['velocity_y'] * vel_scale,
                                    head_width=0.5, head_length=0.8, fc=color, ec=color)
        
        # Update title with timestamp
        timestamp = self.scenario_data['timestamps'][frame_idx]
        plt.title(f"Scenario {self.scenario_data['scenario_id']} - Time: {timestamp:.2f}s")
        
        # Set axis limits with some padding
        all_x = []
        all_y = []
        for track in self.scenario_data['tracks']:
            for state in track['states']:
                if state['valid']:
                    all_x.append(state['center_x'])
                    all_y.append(state['center_y'])
        
        if all_x and all_y:
            padding = 20  # meters
            self.ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
            self.ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
        
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
    
    def create_animation(self, interval: int = 50) -> animation.FuncAnimation:
        """Create an animation of the scenario.
        
        Args:
            interval: Time between frames in milliseconds.
        
        Returns:
            matplotlib.animation.FuncAnimation object
        """
        self.setup_plot()
        num_frames = len(self.scenario_data['timestamps'])
        anim = animation.FuncAnimation(self.fig, self.plot_frame, 
                                     frames=num_frames, interval=interval,
                                     blit=False)
        return anim
    
    def save_animation(self, filename: str, fps: int = 20):
        """Save the animation to a file.
        
        Args:
            filename: Output filename (should end in .mp4)
            fps: Frames per second for the output video
        """
        anim = self.create_animation(interval=1000//fps)
        anim.save(filename, writer='ffmpeg', fps=fps)
    
    def plot_trajectory(self):
        """Plot the full trajectory of all objects in the scenario."""
        self.setup_plot()
        
        # Plot each track's full trajectory
        for track in self.scenario_data['tracks']:
            # Get all valid positions
            x_pos = []
            y_pos = []
            for state in track['states']:
                if state['valid']:
                    x_pos.append(state['center_x'])
                    y_pos.append(state['center_y'])
            
            if x_pos and y_pos:
                color = self.type_colors.get(track['object_type'], 'gray')
                # Plot trajectory line
                self.ax.plot(x_pos, y_pos, '-', color=color, alpha=0.5)
                # Plot start position
                self.ax.scatter(x_pos[0], y_pos[0], c=color, marker='o')
                # Plot end position
                self.ax.scatter(x_pos[-1], y_pos[-1], c=color, marker='s')
        
        plt.show()

def main():
    # Path to the dataset directory
    dataset_dir = "data/waymo/waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training_20s"
    
    # Create output directory for visualizations
    output_dir = "data/waymo/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get tfrecord files
    tfrecord_files = [f for f in os.listdir(dataset_dir) 
                      if "uncompressed_scenario_training_20s_training_20s.tfrecord-" in f]
    if not tfrecord_files:
        print("No tfrecord files found in the specified directory!")
        return
    
    print(f"Found {len(tfrecord_files)} tfrecord files")
    tfrecord_path = os.path.join(dataset_dir, tfrecord_files[0])
    print(f"Reading scenario from: {tfrecord_path}")
    
    # Parse scenario
    reader = WaymoScenarioReader(tfrecord_path)
    scenario_data = reader.parse_scenario()
    
    # Create visualizer
    visualizer = WaymoScenarioVisualizer(scenario_data)
    
    # Create and save animation
    print("Creating animation...")
    animation_path = os.path.join(output_dir, f"scenario_{scenario_data['scenario_id']}_animation.mp4")
    visualizer.save_animation(animation_path, fps=10)
    print(f"Animation saved to: {animation_path}")
    
    # Plot full trajectories
    print("Plotting trajectories...")
    visualizer.plot_trajectory()

if __name__ == "__main__":
    main() 