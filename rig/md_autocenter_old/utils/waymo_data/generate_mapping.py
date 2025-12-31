import os
import json
from pathlib import Path
import glob

def scan_directory(directory, pattern):
    """Scan a directory for files matching a pattern and return their info."""
    files = glob.glob(os.path.join(directory, pattern))
    return [{"path": f, "filename": os.path.basename(f)} for f in files]

def main():
    # Base data directory
    data_dir = os.path.abspath("data/waymo")
    
    # Scan for raw TFRecord files
    tfrecord_dir = os.path.join(data_dir, "waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training_20s")
    tfrecords = scan_directory(tfrecord_dir, "*.tfrecord-*")
    
    # Scan for visualizations
    viz_dir = os.path.join(data_dir, "visualizations")
    visualizations = scan_directory(viz_dir, "scenario_*_animation.mp4")
    
    # Scan for MetaDrive scenarios
    metadrive_dir = os.path.join(data_dir, "metadrive_scenarios")
    metadrive_scenarios = []
    if os.path.exists(metadrive_dir):
        # Read the dataset summary file if it exists
        summary_file = os.path.join(metadrive_dir, "dataset_summary.pkl")
        if os.path.exists(summary_file):
            metadrive_scenarios.append({
                "path": summary_file,
                "filename": "dataset_summary.pkl",
                "type": "summary"
            })
        
        # Add any scenario directories
        scenario_dirs = scan_directory(metadrive_dir, "metadrive_scenarios_*")
        metadrive_scenarios.extend(scenario_dirs)
    
    # Create mapping
    mapping = {
        "raw_data": {
            "directory": tfrecord_dir,
            "files": tfrecords
        },
        "visualizations": {
            "directory": viz_dir,
            "files": visualizations
        },
        "metadrive_scenarios": {
            "directory": metadrive_dir,
            "files": metadrive_scenarios
        },
        "metadata": {
            "created_at": os.path.getmtime(tfrecord_dir),
            "last_updated": os.path.getmtime(metadrive_dir) if os.path.exists(metadrive_dir) else None,
            "num_tfrecords": len(tfrecords),
            "num_visualizations": len(visualizations),
            "num_metadrive_scenarios": len(metadrive_scenarios)
        }
    }
    
    # Save mapping to JSON
    mapping_file = os.path.join(data_dir, "waymo_data_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nMapping file created at: {mapping_file}")
    print("\nSummary:")
    print(f"- TFRecord files: {len(tfrecords)}")
    print(f"- Visualizations: {len(visualizations)}")
    print(f"- MetaDrive scenarios: {len(metadrive_scenarios)}")

if __name__ == "__main__":
    main() 