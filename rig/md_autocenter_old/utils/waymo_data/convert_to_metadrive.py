import os
import sys
from pathlib import Path

def main():
    # Get the absolute path to our Waymo data
    raw_data_path = os.path.abspath("data/waymo/waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training_20s")
    
    # Create output directory for converted scenarios
    output_dir = os.path.abspath("data/waymo/metadrive_scenarios")
    os.makedirs(output_dir, exist_ok=True)
    
    # Disable GPU for conversion (as recommended in docs)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Use ScenarioNet's converter with overwrite flag
    conversion_cmd = f"python -m scenarionet.convert_waymo -d {output_dir} --raw_data_path {raw_data_path} --num_workers 4 --overwrite"
    
    print("Starting Waymo to MetaDrive conversion...")
    print(f"Input directory: {raw_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Running command: {conversion_cmd}")
    
    # Execute the conversion
    os.system(conversion_cmd)
    
    print("\nConversion complete!")
    print(f"Converted scenarios are saved in: {output_dir}")
    print("\nYou can now use these scenarios in MetaDrive using ScenarioNet's API.")

if __name__ == "__main__":
    main() 