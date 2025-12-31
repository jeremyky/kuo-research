# config_gen.py
import argparse, json, os
from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.utils import read_dataset_summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="Path to Waymo-converted dataset (same as -d for data_collection.py)")
    ap.add_argument("--out", type=str, default="configs",
                    help="Where to write config JSONs")
    ap.add_argument("--onset", type=int, default=200,
                    help="Default onset_manual")
    ap.add_argument("--reactive", action="store_true",
                    help="Use reactive_traffic=True in configs (default False for replay-like behavior)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    data_path = args.data or AssetLoader.file_path(AssetLoader.asset_path, "waymo", unix_style=False)
    summary = read_dataset_summary(data_path)

    # MetaDrive returns a tuple; files list is at index 1, as you've been using.
    try:
        files = summary[1]
    except Exception:
        # Fallback in case the API changes someday
        files = summary.get("scenario_files", []) if isinstance(summary, dict) else summary

    count = 0
    for file in files:
        # Example: sd_training.tfrecord-00000-of-01000_2a1e44d405a6833f.pkl
        stem = file.rsplit("/", 1)[-1]
        no_ext = stem.rsplit(".", 1)[0]
        sid = no_ext.split("_")[-1]

        cfg = {
            "instructions": ["follow the lane"],
            "onset_manuals": [args.onset],
            "onset_manual": args.onset,
            "instructions_legacy": "follow the lane",
            "reactive_traffic": bool(args.reactive),
            "extra_steps": 1000
        }
        out_path = os.path.join(args.out, f"{sid}.json")
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Wrote {out_path}")
        count += 1

    print(f"✅ Generated {count} config(s) in {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()



