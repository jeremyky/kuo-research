from reader import WaymoScenarioReader
import os

def main():
    # Path to the dataset directory
    dataset_dir = "data/waymo/waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training_20s"
    
    # Get tfrecord files with the correct naming pattern
    tfrecord_files = [f for f in os.listdir(dataset_dir) if "uncompressed_scenario_training_20s_training_20s.tfrecord-" in f]
    if not tfrecord_files:
        print("No tfrecord files found in the specified directory!")
        return
        
    print(f"Found {len(tfrecord_files)} tfrecord files")
    tfrecord_path = os.path.join(dataset_dir, tfrecord_files[0])
    print(f"Reading scenario from: {tfrecord_path}")
    
    # Initialize reader and parse scenario
    reader = WaymoScenarioReader(tfrecord_path)
    scenario_data = reader.parse_scenario()
    
    # Print summary of the scenario
    reader.print_scenario_summary(scenario_data)

if __name__ == "__main__":
    main() 