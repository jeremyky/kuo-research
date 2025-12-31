#!/usr/bin/env python
"""
Review and delete scenarios marked as bad during data collection.

Usage:
    python review_bad_scenarios.py                    # Review list
    python review_bad_scenarios.py --delete-all       # Delete all marked configs
    python review_bad_scenarios.py --delete-sid <sid> # Delete specific scenario
    python review_bad_scenarios.py --clear            # Clear bad scenarios list
"""

import argparse
import json
import os
import shutil


def load_bad_scenarios(bad_scenarios_path="bad_scenarios.json"):
    """Load the list of scenarios marked for deletion."""
    if not os.path.exists(bad_scenarios_path):
        print(f"No bad scenarios file found at: {bad_scenarios_path}")
        return []
    
    try:
        with open(bad_scenarios_path, "r") as f:
            data = json.load(f)
        return data.get("scenarios", [])
    except Exception as e:
        print(f"❌ Failed to load bad scenarios: {e}")
        return []


def delete_config(sid, config_dir="configs", backup_dir="configs_deleted"):
    """Delete a config file and back it up."""
    config_path = os.path.join(config_dir, f"{sid}.json")
    
    if not os.path.exists(config_path):
        print(f"⚠️  Config not found: {config_path}")
        return False
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, f"{sid}.json")
    
    try:
        # Backup first
        shutil.copy2(config_path, backup_path)
        print(f"📦 Backed up to: {backup_path}")
        
        # Delete original
        os.remove(config_path)
        print(f"🗑️  Deleted: {config_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to delete {sid}: {e}")
        return False


def review_scenarios(bad_scenarios_path="bad_scenarios.json"):
    """Display all scenarios marked as bad."""
    scenarios = load_bad_scenarios(bad_scenarios_path)
    
    if not scenarios:
        print("\n✅ No scenarios marked as bad!")
        return
    
    print(f"\n{'='*80}")
    print(f"BAD SCENARIOS TO REVIEW ({len(scenarios)} total)")
    print(f"{'='*80}\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. Scenario ID: {scenario['sid']}")
        print(f"   Seed: {scenario['seed']}")
        print(f"   Instruction: {scenario['instruction']}")
        print(f"   Reason: {scenario['reason']}")
        print(f"   Marked at: {scenario['marked_at']}")
        print()
    
    print(f"{'='*80}\n")
    print(f"To delete all: python review_bad_scenarios.py --delete-all")
    print(f"To delete one: python review_bad_scenarios.py --delete-sid <sid>")
    print(f"To clear list: python review_bad_scenarios.py --clear")
    print()


def delete_all_scenarios(config_dir="configs", backup_dir="configs_deleted", 
                         bad_scenarios_path="bad_scenarios.json"):
    """Delete all scenarios marked as bad."""
    scenarios = load_bad_scenarios(bad_scenarios_path)
    
    if not scenarios:
        print("✅ No scenarios to delete!")
        return
    
    print(f"\n⚠️  About to delete {len(scenarios)} config files...")
    print(f"Backups will be saved to: {backup_dir}")
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled.")
        return
    
    deleted_count = 0
    for scenario in scenarios:
        if delete_config(scenario['sid'], config_dir, backup_dir):
            deleted_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Deleted {deleted_count}/{len(scenarios)} configs")
    print(f"📦 Backups saved to: {backup_dir}")
    print(f"{'='*60}\n")
    
    # Ask to clear the list
    response = input("Clear the bad scenarios list? (yes/no): ")
    if response.lower() == "yes":
        clear_bad_scenarios_list(bad_scenarios_path)


def delete_single_scenario(sid, config_dir="configs", backup_dir="configs_deleted",
                           bad_scenarios_path="bad_scenarios.json"):
    """Delete a single scenario by ID."""
    scenarios = load_bad_scenarios(bad_scenarios_path)
    
    # Check if in list
    scenario = next((s for s in scenarios if s['sid'] == sid), None)
    if not scenario:
        print(f"⚠️  Scenario {sid} not in bad scenarios list")
        print("Delete anyway? (yes/no): ", end="")
        response = input()
        if response.lower() != "yes":
            return
    
    if delete_config(sid, config_dir, backup_dir):
        print(f"✅ Successfully deleted {sid}")
        
        # Remove from bad list
        if scenario:
            scenarios = [s for s in scenarios if s['sid'] != sid]
            try:
                with open(bad_scenarios_path, "w") as f:
                    json.dump({
                        "scenarios": scenarios,
                        "count": len(scenarios),
                        "note": "Review these scenarios and delete their config files if needed"
                    }, f, indent=2)
                print(f"Updated bad scenarios list ({len(scenarios)} remaining)")
            except Exception as e:
                print(f"⚠️  Failed to update bad scenarios list: {e}")


def clear_bad_scenarios_list(bad_scenarios_path="bad_scenarios.json"):
    """Clear the bad scenarios list."""
    try:
        with open(bad_scenarios_path, "w") as f:
            json.dump({
                "scenarios": [],
                "count": 0,
                "note": "Review these scenarios and delete their config files if needed"
            }, f, indent=2)
        print(f"✅ Cleared bad scenarios list")
    except Exception as e:
        print(f"❌ Failed to clear list: {e}")


def main():
    parser = argparse.ArgumentParser(description="Review and delete bad scenarios")
    parser.add_argument("--delete-all", action="store_true", 
                       help="Delete all scenarios marked as bad")
    parser.add_argument("--delete-sid", type=str, 
                       help="Delete a specific scenario by ID")
    parser.add_argument("--clear", action="store_true",
                       help="Clear the bad scenarios list without deleting configs")
    parser.add_argument("--config-dir", type=str, default="configs",
                       help="Directory containing config files (default: configs)")
    parser.add_argument("--backup-dir", type=str, default="configs_deleted",
                       help="Directory for deleted config backups (default: configs_deleted)")
    parser.add_argument("--bad-list", type=str, default="bad_scenarios.json",
                       help="Path to bad scenarios JSON (default: bad_scenarios.json)")
    
    args = parser.parse_args()
    
    if args.clear:
        clear_bad_scenarios_list(args.bad_list)
    elif args.delete_all:
        delete_all_scenarios(args.config_dir, args.backup_dir, args.bad_list)
    elif args.delete_sid:
        delete_single_scenario(args.delete_sid, args.config_dir, args.backup_dir, args.bad_list)
    else:
        # Default: just review
        review_scenarios(args.bad_list)


if __name__ == "__main__":
    main()

