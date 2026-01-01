#!/usr/bin/env python3
"""
Validate scenario buckets by visualizing scenarios from each bucket.

This script lets you run scenarios from the same bucket in MetaDrive
to visually confirm they are similar.

Usage:
    # Validate bucket 0 (run 3 random scenarios from bucket 0)
    python validate_buckets.py --buckets data/waymo_buckets.json \
                                 --scenario_dir ../drive-rig/datasets/waymo_converted_test/waymo_converted_test_0/ \
                                 --bucket_id 0 \
                                 --num_scenarios 3
    
    # Compare two buckets side-by-side
    python validate_buckets.py --buckets data/waymo_buckets.json \
                                 --scenario_dir ../drive-rig/datasets/waymo_converted_test/waymo_converted_test_0/ \
                                 --bucket_id 0,5 \
                                 --num_scenarios 2
"""

import json
import pickle
import random
import sys
from pathlib import Path

# NumPy 2.0+ compatibility fix for pickle files created with older numpy
import numpy as np
if not hasattr(np, '_core'):
    np._core = np.core
    sys.modules['numpy._core'] = np.core
if not hasattr(np, '_core._multiarray_umath'):
    try:
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
    except AttributeError:
        pass


def load_bucketing(json_path: str):
    """Load bucketing results."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_scenarios_from_bucket(bucketing: dict, bucket_id: int, num_scenarios: int = 3):
    """Get random scenarios from a bucket."""
    bucket_data = bucketing['clustering']['buckets'][str(bucket_id)]
    scenarios = bucket_data['scenarios']
    
    if num_scenarios > len(scenarios):
        print(f"Warning: Bucket {bucket_id} only has {len(scenarios)} scenarios, using all")
        return scenarios
    
    return random.sample(scenarios, num_scenarios)


def load_and_run_scenario(scenario_path: str, render: bool = True, image_dir: Path = None, 
                          scenario_metadata: dict = None):
    """
    Load a Waymo scenario and replay it using ReplayEgoCarPolicy.
    Captures screenshots of initial position and trajectory.
    
    Args:
        scenario_path: Path to .pkl file
        render: Whether to render visually
        image_dir: Directory to save screenshots
        scenario_metadata: Metadata dict with bucket info, clustering info, etc.
    """
    try:
        import sys
        import pygame
        from pathlib import Path
        
        # Initialize pygame for keyboard events
        pygame.init()
        
        scenario_path = Path(scenario_path)
        scenario_dir = scenario_path.parent
        scenario_id = scenario_path.stem
        
        # Add rig metadrive to path
        rig_metadrive = Path(__file__).parent.parent.parent / "rig" / "metadrive"
        if rig_metadrive.exists():
            sys.path.insert(0, str(rig_metadrive))
        
        from metadrive.envs.custom_scenario_env import CustomScenarioEnv
        from metadrive.policy.replay_policy import ReplayEgoCarPolicy
        from metadrive.scenario.utils import read_dataset_summary
        from metadrive.engine.engine_utils import get_engine
        
        # Find scenario index
        summary = read_dataset_summary(str(scenario_dir))
        try:
            files = summary[1] if isinstance(summary, tuple) else summary.get("scenario_files", [])
        except:
            files = list(scenario_dir.glob("*.pkl"))
            files = [str(f) for f in sorted(files)]
        
        try:
            scenario_index = next(i for i, f in enumerate(files) if scenario_path.name in str(f))
        except StopIteration:
            scenario_index = 0
        
        print(f"\n{'='*80}")
        print(f"Running: {scenario_path.name}")
        print(f"{'='*80}")
        
        # Use same config as data_collection.py replay function
        config = {
            "manual_control": False,
            "reactive_traffic": False,
            "use_render": render,
            "agent_policy": ReplayEgoCarPolicy,  # Replays original Waymo trajectory
            "data_directory": str(scenario_dir),
            "start_scenario_index": scenario_index,
            "num_scenarios": 1,
            "window_size": (1920, 1080),
            "camera_dist": 10.0,
            "camera_height": 3.5,
            "camera_fov": 80,
        }
        
        env = CustomScenarioEnv(config)
        obs, info = env.reset()
        
        # Set up keyboard handlers using Panda3D messenger
        engine = get_engine()
        next_scenario = False
        quit_requested = False
        
        def on_next_key():
            nonlocal next_scenario
            next_scenario = True
            print("\n  → Next scenario requested (N key pressed)")
        
        def on_quit_key():
            nonlocal quit_requested
            quit_requested = True
            print("\n  → Quit requested (Q key pressed)")
        
        # Register keyboard handlers with Panda3D
        engine.accept("n", on_next_key)
        engine.accept("N", on_next_key)
        engine.accept("q", on_quit_key)
        engine.accept("Q", on_quit_key)
        engine.accept("escape", on_quit_key)
        
        # Initialize topdown renderer first
        if image_dir:
            image_dir.mkdir(parents=True, exist_ok=True)
            # Initialize renderer
            env.render(
                mode="topdown",
                screen_size=(1600, 1600),
                film_size=(2000, 2000),
                target_agent_heading_up=True,  # Fixed deprecated parameter
                semantic_map=True,
            )
        
        # Wait for scene to fully load (let MetaDrive finish rendering)
        import time
        print("Waiting for scene to load...")
        for _ in range(20):  # ~2 seconds at 60fps, or use time.sleep
            engine.taskMgr.step()
            if render:
                env.render()
            time.sleep(0.1)  # 100ms per step = 2 seconds total
        
        # Capture initial position screenshots
        initial_3d_path = None
        initial_topdown_path = None
        if image_dir:
            try:
                # Small delay after loading to ensure rendering is stable
                time.sleep(0.5)
                
                # Capture 3D view
                initial_3d_path = image_dir / f"{scenario_id}_initial_3d.png"
                env.capture(str(initial_3d_path))
                print(f"  ✓ Saved initial 3D view: {initial_3d_path.name}")
                
                # Small delay between screenshots
                time.sleep(0.3)
            except Exception as e:
                print(f"  ⚠️  Could not save initial 3D screenshot: {e}")
                import traceback
                traceback.print_exc()
            
            try:
                # Render top-down view for initial position
                # Use target_agent_heading_up instead of deprecated target_vehicle_heading_up
                topdown_img = env.render(
                    mode="topdown",
                    screen_size=(1600, 1600),
                    film_size=(2000, 2000),
                    target_agent_heading_up=True,  # Fixed deprecated parameter
                    semantic_map=True,
                    to_image=False,  # Returns pygame surface
                )
                if topdown_img is not None:
                    initial_topdown_path = image_dir / f"{scenario_id}_initial_topdown.png"
                    pygame.image.save(topdown_img, str(initial_topdown_path))
                    print(f"  ✓ Saved initial topdown: {initial_topdown_path.name}")
            except Exception as e:
                print(f"  ⚠️  Could not save initial topdown screenshot: {e}")
                import traceback
                traceback.print_exc()
        
        print("Replaying scenario... (Press 'N' for next, 'Q' to quit)")
        done = False
        truncated = False
        step = 0
        
        # Run scenario to completion with keyboard controls
        while not (done or truncated) and step < 2000 and not next_scenario and not quit_requested:
            # Process Panda3D events
            engine.taskMgr.step()
            
            if next_scenario or quit_requested:
                break
            
            action = [0.0, 0.0]
            obs, reward, done, truncated, info = env.step(action)
            
            # Render normally
            if render:
                env.render()
            
            step += 1
        
        # Capture final trajectory screenshots
        trajectory_3d_path = None
        trajectory_topdown_path = None
        if image_dir and not quit_requested:
            try:
                # Capture 3D view
                trajectory_3d_path = image_dir / f"{scenario_id}_trajectory_3d.png"
                env.capture(str(trajectory_3d_path))
                print(f"  ✓ Saved trajectory 3D view: {trajectory_3d_path.name}")
            except Exception as e:
                print(f"  ⚠️  Could not save trajectory 3D screenshot: {e}")
            
            try:
                # Small delay before topdown screenshot
                time.sleep(0.3)
                
                topdown_img = env.render(
                    mode="topdown",
                    screen_size=(1600, 1600),
                    film_size=(2000, 2000),
                    target_agent_heading_up=True,  # Fixed deprecated parameter
                    semantic_map=True,
                    to_image=False,  # Returns pygame surface
                )
                if topdown_img is not None:
                    trajectory_topdown_path = image_dir / f"{scenario_id}_trajectory_topdown.png"
                    pygame.image.save(topdown_img, str(trajectory_topdown_path))
                    print(f"  ✓ Saved trajectory topdown: {trajectory_topdown_path.name}")
            except Exception as e:
                print(f"  ⚠️  Could not save trajectory topdown screenshot: {e}")
                import traceback
                traceback.print_exc()
        
        # Save metadata JSON
        if image_dir and scenario_metadata:
            metadata_path = image_dir / f"{scenario_id}_metadata.json"
            metadata = {
                "scenario_id": scenario_id,
                "scenario_path": str(scenario_path),
                "images": {
                    "initial_3d": str(initial_3d_path.name) if initial_3d_path else None,
                    "initial_topdown": str(initial_topdown_path.name) if initial_topdown_path else None,
                    "trajectory_3d": str(trajectory_3d_path.name) if trajectory_3d_path else None,
                    "trajectory_topdown": str(trajectory_topdown_path.name) if trajectory_topdown_path else None,
                },
                **scenario_metadata
            }
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  ✓ Saved metadata: {metadata_path.name}")
        
        if quit_requested:
            print(f"\nQuitting...")
            try:
                env.close()
            except:
                pass
            return {'success': False, 'quit': True, 'steps': step}
        
        print(f"\nFinished: {step} steps")
        
        # Close environment properly - this should close the window
        try:
            env.close()
        except:
            pass
        
        # Small delay to ensure cleanup
        import time
        time.sleep(0.3)
        
        return {'success': True, 'steps': step, 'next': next_scenario}
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return {'success': False, 'quit': True}
    except ImportError as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_bucket_info(bucketing: dict, bucket_id: int):
    """Print information about a bucket."""
    bucket_data = bucketing['clustering']['buckets'][str(bucket_id)]
    
    print(f"\n{'='*80}")
    print(f"BUCKET {bucket_id} INFORMATION")
    print(f"{'='*80}")
    print(f"Number of scenarios: {bucket_data['size']}")
    print(f"Average route length: {bucket_data['avg_length_m']:.1f}m")
    print(f"Average curvature: {bucket_data['avg_curvature']:.4f}")
    print(f"Average vehicles: {bucket_data['avg_num_vehicles']:.1f}")
    print(f"Average density: {bucket_data['avg_density']:.2f} vehicles/100m")
    print(f"Common maneuvers: {', '.join([f'{tok}({cnt})' for tok, cnt in bucket_data['common_tokens']])}")
    print(f"{'='*80}\n")


def compare_buckets_stats(bucketing: dict, bucket_ids: list):
    """Print comparison of multiple buckets."""
    print(f"\n{'='*80}")
    print(f"BUCKET COMPARISON")
    print(f"{'='*80}\n")
    
    stats = []
    for bid in bucket_ids:
        bucket_data = bucketing['clustering']['buckets'][str(bid)]
        stats.append({
            'id': bid,
            'size': bucket_data['size'],
            'length': bucket_data['avg_length_m'],
            'curvature': bucket_data['avg_curvature'],
            'vehicles': bucket_data['avg_num_vehicles'],
            'density': bucket_data['avg_density'],
        })
    
    # Print table
    print(f"{'Bucket':<8} {'Size':<8} {'Length(m)':<12} {'Curvature':<12} {'Vehicles':<10} {'Density':<10}")
    print("-" * 80)
    for s in stats:
        print(f"{s['id']:<8} {s['size']:<8} {s['length']:<12.1f} {s['curvature']:<12.4f} {s['vehicles']:<10.1f} {s['density']:<10.2f}")
    
    print("\n" + "="*80 + "\n")


def interactive_validation(bucketing: dict, scenario_dir: str):
    """Interactive mode to explore buckets."""
    print("\n" + "="*80)
    print("INTERACTIVE BUCKET VALIDATION")
    print("="*80)
    print("\nCommands:")
    print("  info <bucket_id>        - Show bucket information")
    print("  compare <id1>,<id2>     - Compare two buckets")
    print("  run <bucket_id> <n>     - Run n scenarios from bucket")
    print("  list                     - List all buckets")
    print("  quit                     - Exit")
    print("="*80)
    
    while True:
        try:
            cmd = input("\n> ").strip().split()
            if not cmd:
                continue
            
            action = cmd[0].lower()
            
            if action == 'quit':
                break
            
            elif action == 'list':
                for bucket_id in sorted(bucketing['clustering']['buckets'].keys()):
                    print_bucket_info(bucketing, int(bucket_id))
            
            elif action == 'info' and len(cmd) > 1:
                bucket_id = int(cmd[1])
                print_bucket_info(bucketing, bucket_id)
            
            elif action == 'compare' and len(cmd) > 1:
                bucket_ids = [int(x.strip()) for x in cmd[1].split(',')]
                compare_buckets_stats(bucketing, bucket_ids)
            
            elif action == 'run' and len(cmd) > 2:
                bucket_id = int(cmd[1])
                num_scenarios = int(cmd[2])
                
                print_bucket_info(bucketing, bucket_id)
                scenarios = get_scenarios_from_bucket(bucketing, bucket_id, num_scenarios)
                
                for i, scenario_info in enumerate(scenarios, 1):
                    print(f"\n--- Scenario {i}/{len(scenarios)} ---")
                    scenario_path = scenario_info['file_path']
                    if not Path(scenario_path).exists():
                        # Try relative to scenario_dir
                        scenario_path = str(Path(scenario_dir) / Path(scenario_path).name)
                    
                    result = load_and_run_scenario(scenario_path, render=True)
                    
                    if i < len(scenarios):
                        input("Press Enter to run next scenario...")
            
            else:
                print("Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='Validate scenario buckets by running scenarios',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--buckets', type=str, required=True,
                        help='Bucketing JSON file')
    parser.add_argument('--scenario_dir', type=str, 
                        default='../data/waymo/converted/waymo_converted_test_0/',
                        help='Directory with scenario .pkl files')
    parser.add_argument('--bucket_id', type=str, default=None,
                        help='Bucket ID(s) to validate (comma-separated for comparison)')
    parser.add_argument('--num_scenarios', type=int, default=3,
                        help='Number of scenarios to run per bucket')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode')
    parser.add_argument('--save_images', action='store_true',
                        help='Save screenshots of initial position and trajectory')
    parser.add_argument('--image_dir', type=str, default='data/waymo/buckets/images',
                        help='Directory to save screenshots')
    
    args = parser.parse_args()
    
    # Load bucketing
    print(f"Loading bucketing from {args.buckets}...")
    bucketing = load_bucketing(args.buckets)
    print(f"✓ Loaded {bucketing['metadata']['n_buckets']} buckets with {bucketing['metadata']['n_scenarios']} scenarios")
    
    if args.interactive:
        interactive_validation(bucketing, args.scenario_dir)
    
    elif args.bucket_id:
        bucket_ids = [int(x.strip()) for x in args.bucket_id.split(',')]
        
        if len(bucket_ids) > 1:
            # Compare mode
            compare_buckets_stats(bucketing, bucket_ids)
            
            for bucket_id in bucket_ids:
                print(f"\n{'#'*80}")
                print(f"# Running scenarios from Bucket {bucket_id}")
                print(f"{'#'*80}")
                
                print_bucket_info(bucketing, bucket_id)
                scenarios = get_scenarios_from_bucket(bucketing, bucket_id, args.num_scenarios)
                
                # Create organized directory structure
                if args.save_images:
                    base_dir = Path(args.image_dir)
                    clustering_type = bucketing.get('metadata', {}).get('clustering_type', 'kmeans')
                    signature_method = bucketing.get('metadata', {}).get('signature_method', 'default')
                    image_dir = base_dir / clustering_type / signature_method / f"bucket_{bucket_id}"
                else:
                    image_dir = None
                
                for i, scenario_info in enumerate(scenarios, 1):
                    print(f"\n--- Bucket {bucket_id}, Scenario {i}/{len(scenarios)} ---")
                    scenario_path = scenario_info['file_path']
                    if not Path(scenario_path).exists():
                        scenario_path = str(Path(args.scenario_dir) / Path(scenario_path).name)
                    
                    # Prepare metadata
                    bucket_data = bucketing['clustering']['buckets'][str(bucket_id)]
                    scenario_metadata = {
                        "bucket_id": bucket_id,
                        "scenario_index_in_bucket": i - 1,
                        "clustering": {
                            "type": bucketing.get('metadata', {}).get('clustering_type', 'kmeans'),
                            "n_clusters": bucketing.get('metadata', {}).get('n_clusters', 'unknown'),
                            "silhouette_score": bucketing.get('clustering', {}).get('silhouette_score', None),
                        },
                        "signature_method": bucketing.get('metadata', {}).get('signature_method', 'default'),
                        "bucket_stats": {
                            "size": bucket_data.get('size', 0),
                            "avg_length_m": bucket_data.get('avg_length_m', 0),
                            "avg_curvature": bucket_data.get('avg_curvature', 0),
                            "avg_num_vehicles": bucket_data.get('avg_num_vehicles', 0),
                            "avg_density": bucket_data.get('avg_density', 0),
                        },
                        "scenario_features": scenario_info.get('features', {}),
                    }
                    
                    result = load_and_run_scenario(scenario_path, render=True, image_dir=image_dir, 
                                                  scenario_metadata=scenario_metadata)
                    
                    # Check if user wants to quit
                    if result and result.get('quit'):
                        sys.exit(0)
                    
                    # If not auto-advancing, wait for next key
                    if not result or not result.get('next'):
                        if i < len(scenarios):
                            print("\nPress 'N' or Enter to continue to next scenario...")
                            # Wait for key press
                            waiting = True
                            while waiting:
                                for event in pygame.event.get():
                                    if event.type == pygame.KEYDOWN:
                                        if event.key in (pygame.K_n, pygame.K_RETURN, pygame.K_q, pygame.K_ESCAPE):
                                            waiting = False
                                            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                                                sys.exit(0)
                                            break
                                import time
                                time.sleep(0.1)
                
                if bucket_id != bucket_ids[-1]:
                    print("\nPress 'N' or Enter to continue to next bucket...")
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key in (pygame.K_n, pygame.K_RETURN, pygame.K_q, pygame.K_ESCAPE):
                                    waiting = False
                                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                                        sys.exit(0)
                                    break
                        import time
                        time.sleep(0.1)
        
        else:
            # Single bucket mode
            bucket_id = bucket_ids[0]
            print_bucket_info(bucketing, bucket_id)
            scenarios = get_scenarios_from_bucket(bucketing, bucket_id, args.num_scenarios)
            
            print(f"\nRunning {len(scenarios)} scenarios from bucket {bucket_id}...")
            print("Watch for similarities in:")
            print("  - Road geometry (straight, curved, intersections)")
            print("  - Traffic density")
            print("  - Maneuver types")
            print()
            
            # Create organized directory structure
            if args.save_images:
                base_dir = Path(args.image_dir)
                clustering_type = bucketing.get('metadata', {}).get('clustering_type', 'kmeans')
                signature_method = bucketing.get('metadata', {}).get('signature_method', 'default')
                image_dir = base_dir / clustering_type / signature_method / f"bucket_{bucket_id}"
            else:
                image_dir = None
            
            for i, scenario_info in enumerate(scenarios, 1):
                print(f"\n--- Scenario {i}/{len(scenarios)} ---")
                scenario_path = scenario_info['file_path']
                if not Path(scenario_path).exists():
                    scenario_path = str(Path(args.scenario_dir) / Path(scenario_path).name)
                
                # Prepare metadata
                bucket_data = bucketing['clustering']['buckets'][str(bucket_id)]
                scenario_metadata = {
                    "bucket_id": bucket_id,
                    "scenario_index_in_bucket": i - 1,
                    "clustering": {
                        "type": bucketing.get('metadata', {}).get('clustering_type', 'kmeans'),
                        "n_clusters": bucketing.get('metadata', {}).get('n_clusters', 'unknown'),
                        "silhouette_score": bucketing.get('clustering', {}).get('silhouette_score', None),
                    },
                    "signature_method": bucketing.get('metadata', {}).get('signature_method', 'default'),
                    "bucket_stats": {
                        "size": bucket_data.get('size', 0),
                        "avg_length_m": bucket_data.get('avg_length_m', 0),
                        "avg_curvature": bucket_data.get('avg_curvature', 0),
                        "avg_num_vehicles": bucket_data.get('avg_num_vehicles', 0),
                        "avg_density": bucket_data.get('avg_density', 0),
                    },
                    "scenario_features": scenario_info.get('features', {}),
                }
                
                result = load_and_run_scenario(scenario_path, render=True, image_dir=image_dir,
                                              scenario_metadata=scenario_metadata)
                
                # Check if user wants to quit
                if result and result.get('quit'):
                    sys.exit(0)
                
                # If not auto-advancing, wait for next key
                if not result or not result.get('next'):
                    if i < len(scenarios):
                        print("\nPress 'N' or Enter to continue to next scenario...")
                        # Wait for key press
                        waiting = True
                        while waiting:
                            for event in pygame.event.get():
                                if event.type == pygame.KEYDOWN:
                                    if event.key in (pygame.K_n, pygame.K_RETURN, pygame.K_q, pygame.K_ESCAPE):
                                        waiting = False
                                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                                            sys.exit(0)
                                        break
                            import time
                            time.sleep(0.1)
            
            print("\n✓ Validation complete!")
            print("\nDid the scenarios look similar? If not:")
            print("  - Try different n_clusters")
            print("  - Check if you have enough scenarios")
            print("  - Adjust feature weights in bucketing code")
    
    else:
        print("\nNo bucket specified. Available buckets:")
        for bucket_id in sorted(bucketing['clustering']['buckets'].keys()):
            stats = bucketing['clustering']['buckets'][bucket_id]
            print(f"  Bucket {bucket_id}: {stats['size']} scenarios")
        print("\nUsage:")
        print(f"  python validate_buckets.py --buckets {args.buckets} --scenario_dir {args.scenario_dir} --bucket_id 0")
        print(f"  python validate_buckets.py --buckets {args.buckets} --scenario_dir {args.scenario_dir} --interactive")

