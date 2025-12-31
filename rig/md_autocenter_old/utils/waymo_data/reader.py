import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
import numpy as np
from typing import Dict, List, Optional

"""
The reader extracts information from Waymo Motion Dataset scenarios including:
- Scenario ID
- Timestamps (in seconds)
- Object tracks (positions, velocities, types)
- Map features and dynamic states
"""

class WaymoScenarioReader:
    def __init__(self, tfrecord_path: str):
        """Initialize the reader with a tfrecord file path."""
        self.tfrecord_path = tfrecord_path
        raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
        self.dataset = raw_dataset.take(1)  # Take only the first scenario

    def parse_scenario(self) -> Dict:
        """Parse a single scenario from the tfrecord file.
        
        The data is stored as a serialized Scenario protobuf message.
        See scenario.proto for the full message definition.
        """
        for raw_data in self.dataset:
            # Parse the raw bytes as a Scenario proto
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw_data.numpy())
            
            # Extract basic scenario information
            scenario_data = {
                'scenario_id': scenario.scenario_id,
                'timestamps': list(scenario.timestamps_seconds),
                'current_time_index': scenario.current_time_index,
                'sdc_track_index': scenario.sdc_track_index,
                'tracks': []
            }
            
            # Process each track
            for track in scenario.tracks:
                track_dict = {
                    'id': track.id,
                    'object_type': track.object_type,
                    'states': []
                }
                
                # Process each state in the track
                for state in track.states:
                    state_dict = {
                        'center_x': state.center_x,
                        'center_y': state.center_y,
                        'velocity_x': state.velocity_x,
                        'velocity_y': state.velocity_y,
                        'valid': state.valid
                    }
                    track_dict['states'].append(state_dict)
                
                scenario_data['tracks'].append(track_dict)
            
            return scenario_data
            
        return {}  # Return empty dict if no scenario found

    def print_scenario_summary(self, scenario_data: Dict):
        """Print a human-readable summary of the scenario."""
        if not scenario_data:
            print("No scenario data found")
            return
            
        print("\nScenario Summary:")
        print(f"ID: {scenario_data['scenario_id']}")
        print(f"Duration: {len(scenario_data['timestamps'])} timestamps")
        
        if scenario_data['timestamps']:
            print(f"Time range: {min(scenario_data['timestamps']):.2f}s to {max(scenario_data['timestamps']):.2f}s")
            print(f"Current time index: {scenario_data['current_time_index']}")
            print(f"SDC track index: {scenario_data['sdc_track_index']}")
            
            # Track statistics
            n_tracks = len(scenario_data['tracks'])
            print(f"\nNumber of tracks: {n_tracks}")
            
            if n_tracks > 0:
                # Get statistics about object movements
                all_x = []
                all_y = []
                all_vx = []
                all_vy = []
                
                for track in scenario_data['tracks']:
                    for state in track['states']:
                        if state['valid']:
                            all_x.append(state['center_x'])
                            all_y.append(state['center_y'])
                            all_vx.append(state['velocity_x'])
                            all_vy.append(state['velocity_y'])
                
                if all_x:  # Only print if we have valid states
                    print("\nMovement Statistics:")
                    print(f"X position range: {min(all_x):.2f} to {max(all_x):.2f} meters")
                    print(f"Y position range: {min(all_y):.2f} to {max(all_y):.2f} meters")
                    velocities = np.sqrt(np.array(all_vx)**2 + np.array(all_vy)**2)
                    print(f"Max velocity: {max(velocities):.2f} m/s") 