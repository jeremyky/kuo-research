import os

data = "/datasets/waymo_converted_test/waymo_converted_test_0"
cfg_dir = "../configs"

# scenario IDs from PKL filenames: sd_waymo_v1.2_<ID>.pkl  ->  <ID>
scenario_files = [f for f in os.listdir(data) if f.endswith(".pkl") and f.startswith("sd_waymo")]
scenario_ids = {os.path.splitext(f)[0].split("sd_waymo_v1.2_")[-1] for f in scenario_files}

# config IDs from JSON basenames: <ID>.json  ->  <ID>
cfg_ids = {os.path.splitext(f)[0] for f in os.listdir(cfg_dir) if f.endswith(".json")}

print("Scenarios:", len(scenario_ids), "Configs:", len(cfg_ids))
missing = sorted(scenario_ids - cfg_ids)
extra   = sorted(cfg_ids - scenario_ids)
print("Missing configs:", len(missing), " (first 10):", missing[:10])
print("Configs with no matching scenario:", len(extra), " (first 10):", extra[:10])



