import argparse
import json
import os
import random
from metadrive.engine.engine_utils import get_global_config, get_engine
from metadrive.engine.asset_loader import AssetLoader
from partial_manual_policy import HybridEgoCarPolicy, HybridEgoCarReplayPolicy
from metadrive.envs.custom_scenario_env import CustomScenarioEnv
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.scenario.utils import read_dataset_summary, read_scenario_data
from gymnasium.spaces import Discrete
from PIL import Image


# os.environ["SDL_VIDEODRIVER"] = "dummy"  # Hide the pygame window
waymo_data = AssetLoader.file_path(AssetLoader.asset_path, "waymo", unix_style=False)

MAX_STEPS = 100000

HybridEgoCarPolicy.set_extra_input_space(Discrete(MAX_STEPS))
HybridEgoCarReplayPolicy.set_extra_input_space(Discrete(MAX_STEPS))

'''
def _load_configs(config_paths):
    """Return dict[id -> config_json]."""
    return {c.split(".")[0].split("/")[1]: json.load(open(c)) for c in config_paths}
'''

def _load_configs(config_paths):
    """Return dict[id -> config_json]."""
    out = {}
    for c in config_paths:
        if not c.endswith(".json"):
            continue
        key = os.path.splitext(os.path.basename(c))[0]  # <-- robust
        with open(c, "r") as f:
            out[key] = json.load(f)
    return out


def _pick_instruction_and_onset(cfg):
    """
    Choose one instruction/onset from a config that may contain lists or singletons.
    Returns (instruction_str, onset_step_int, index_used_int).
    """
    instr_list = cfg.get("instructions")
    onset_list = cfg.get("onset_manuals")
    if isinstance(instr_list, list) and len(instr_list) > 0:
        idx = random.randrange(len(instr_list))
        instr = instr_list[idx]
        # If onset list is present and long enough, align by index; else fallback
        if isinstance(onset_list, list) and idx < len(onset_list):
            onset = int(onset_list[idx])
        else:
            onset = int(cfg.get("onset_manual", 0))
        return str(instr), onset, idx

    # Backward-compat single fields
    instr = cfg.get("instructions") or cfg.get("instructions_legacy") or "follow the lane"
    onset = int(cfg.get("onset_manual", 0))
    return str(instr), onset, 0


def replay(configs, data, controller="keyboard"):
    """Replays the scenes using the manually generated input when applicable."""
    configs = _load_configs(configs)
    dataset_summary = read_dataset_summary(data)
    num_scenarios = len(dataset_summary[1])

    env = CustomScenarioEnv(
        {
            "manual_control": False,
            "controller": controller,
            "reactive_traffic": False,
            "use_render": True,
            "agent_policy": HybridEgoCarReplayPolicy,
            "data_directory": data,
            "num_scenarios": num_scenarios,
            "allowed_more_steps": 10000,
        }
    )

    for seed in range(num_scenarios):
        
        # remove the prefilter block:
        # if data != waymo_data:
        #     sid_hint = dataset_summary[1][seed].split("_")[3].split(".")[0]
        #     if sid_hint not in configs:
        #         continue

        # Reset, get the *real* id, then filter
        o, _ = env.reset(seed=seed)
        engine = get_engine()
        sid = engine.data_manager.current_scenario["id"]
        if sid not in configs:
            continue
        cfg = configs[sid]







        # Reset scenario
        o, _ = env.reset(seed=seed)
        engine = get_engine()
        sid = engine.data_manager.current_scenario["id"]
        cfg = configs[sid]
        '''
                # NEW: resume wheel haptics only now (this is the reset you actually run)
        try:
            if hasattr(env, "agent") and hasattr(env.agent, "policy"):
                ctrl = getattr(env.agent.policy, "controller", None)
                if hasattr(ctrl, "resume_autocenter"):
                    ctrl.resume_autocenter(delay_s=0.5)
        except Exception as e:
            print(f"⚠️ could not resume wheel for new scenario (replay): {e}")
        '''
# hard zero the shared inputs so the spring targets true 0 from frame 1
        try:
            if hasattr(env, "agent") and hasattr(env.agent, "policy"):
                ctrl = getattr(env.agent.policy, "controller", None)
                if ctrl and hasattr(ctrl, "_lock"):
                    with ctrl._lock:
                        ctrl._shared["steering"] = 0.0
                        ctrl._shared["speed_kmh"] = 0.0
                if hasattr(ctrl, "prepare_for_new_scenario"):
                    # re-init devices cleanly + arm a zero-lock window (explained below)
                    ctrl.prepare_for_new_scenario(device_reset_time_s=0.5, center_lock_s=1.0, spring_boost=1.0)
        except Exception as e:
            print(f"⚠️ wheel re-center prep failed: {e}")


        # Use the SAME filename as 'collect'
        input_path = f"manual_inputs/{sid}.json"

        # Load the exact instruction/onset chosen during collection (if present)
        selected_onset = None
        selected_instr = None
        try:
            with open(input_path, "r") as fh:
                mi = json.load(fh)
            selected_onset = int(mi.get("selected_onset_manual")) if "selected_onset_manual" in mi else None
            selected_instr = mi.get("selected_instruction")
        except FileNotFoundError:
            pass
        except Exception:
            # Ignore malformed files and fall back deterministically
            selected_onset = None
            selected_instr = None

        # Deterministic fallbacks (no randomness in replay)
        if selected_onset is None:
            if isinstance(cfg.get("onset_manuals"), list) and cfg["onset_manuals"]:
                selected_onset = int(cfg["onset_manuals"][0])
            else:
                selected_onset = int(cfg.get("onset_manual", 0))

        if selected_instr is None:
            if isinstance(cfg.get("instructions"), list) and cfg["instructions"]:
                selected_instr = str(cfg["instructions"][0])
            else:
                selected_instr = str(cfg.get("instructions") or cfg.get("instructions_legacy") or "follow the lane")

        global_config = get_global_config()
        global_config.update(
            {
                "onset_manual": selected_onset,
                "manual_inputs": input_path,  # exact path that collect wrote
                "replay_control": False,
                "allowed_more_steps": cfg.get("extra_steps", 1),
                "reactive_traffic": cfg.get("reactive_traffic", False),
                "trajectory_dots": False
            }
        )

        screen_message = ScreenMessage()
        screen_message.POS = (-0.25, 0.75)

        for i in range(MAX_STEPS):
            # action is ignored; 'extra' is used as index
            o, r, tm, tc, info = env.step({"action": [0, 0], "extra": i})

            global_config = get_global_config()
            message = (
                f"Replaying Human Inputs\nInstruction: {selected_instr}"
                if global_config["replay_control"]
                else "Replaying Original Trajectory"
            )
            screen_message.render(message)
            if tm or tc:
                screen_message.clear()
                break

    env.close()
    return



def collect(configs, data, controller="keyboard"):
    """Collects manual input according to given configs and scenes."""
    configs = _load_configs(configs)
    dataset_summary = read_dataset_summary(data)
    num_scenarios = len(dataset_summary[1])
    
    non_executables = []

    env = CustomScenarioEnv(
        {
            "manual_control": False,
            "controller": controller,
            "reactive_traffic": True,
            "use_render": True,
            "agent_policy": HybridEgoCarPolicy,
            "data_directory": data,
            "num_scenarios": num_scenarios,
            "allowed_more_steps": 1,
            
        }
    )

    engine = get_engine()
    for seed in range(num_scenarios):
        if data != waymo_data:
            sid = dataset_summary[1][seed].split("_")[3].split(".")[0]
            if sid not in configs:
                continue


        # Reset everything for start of scenario
        sampled = False
        
        steering_signals, accelerations = [], []
        positions, headings, velocities = [], [], []
        
        o, _ = env.reset(seed=seed)
        engine = get_engine()
        global_config = get_global_config()


        # NEW: Reset wheel autocenter between scenarios (only in collect; replay has no controller)
        try:
            # HybridEgoCarPolicy owns .controller when manual_control/use_render are set
            if hasattr(env, "agent") and hasattr(env.agent, "policy"):
                ctrl = getattr(env.agent.policy, "controller", None)
                if hasattr(ctrl, "reset_between_scenarios"):
                    ctrl.reset_between_scenarios(delay_s=0.6)
        except Exception as e:
            print(f"⚠️ could not reset wheel for new scenario: {e}")







        sid = engine.data_manager.current_scenario["id"]
        if sid not in configs:
            continue
        cfg = configs[sid]
        # Randomly select one instruction/onset for THIS collection run
        selected_instr, selected_onset, selected_idx = _pick_instruction_and_onset(cfg)
        
        skipped = False
        def skip():
            nonlocal skipped
            skipped = not skipped
            non_executables.append((sid, selected_instr, selected_onset, selected_idx))
            
        passed = False
        def con():
            nonlocal passed
            passed = not passed
            
        def turn_off_trajectory():
            global_config.update({
                "trajectory_dots": False
            })
            
        engine.accept("x", skip)
        engine.accept("z", con)
        engine.accept("m", turn_off_trajectory)

        global_config.update(
            {
                "onset_manual": selected_onset,
                "allowed_more_steps": cfg.get("extra_steps", 1),
                "reactive_traffic": cfg.get("reactive_traffic", False),
                "takeover": False,
            }
        )
        

        screen_message = ScreenMessage()
        screen_message.POS = (-0.25, 0.75)

        # Where to write the collected manual inputs
        output_path = f"manual_inputs/{sid}.json"
        agent = env.agent

        for i in range(MAX_STEPS):
            try:
                # action is ignored; 'extra' is used as index
                o, r, tm, tc, info = env.step({"action": [0, 0], "extra": i})
            except Exception:
                screen_message.clear()
                print("Error with current scenario. Continuing to next one")
                break

            global_config = get_global_config()
            

            if global_config["takeover"]:
                if not sampled and i > global_config['onset_manual'] + 5:
                    sampled = True
                    frame = engine._get_window_image()
                    os.makedirs(f'samples/samples_{seed}', exist_ok=True)
                    Image.fromarray(frame).save(f'samples/samples_{seed}/sample_index_{i}.png')
                if info["raw_action"]:
                    steering_signals.append(info["raw_action"][0])
                    accelerations.append(info["raw_action"][1])
                else:
                    steering_signals.append(0)
                    accelerations.append(0)
                positions.append(agent.position.tolist())
                headings.append(agent.heading.tolist())
                velocities.append(agent.velocity.tolist())

                message = (
                    f"{selected_instr}\n"
                    " Use W and S for acceleration and braking respectively,\n"
                    " and A and D for turning left and right.\n"
                    " Press X if you find the instructions impossible to execute,\n"
                    " and Z if you finish the instructions early."
                )
            else:
                message = (
                    "Currently playing loaded trajectory.\n"
                    " Prepare to take over shortly.\n"
                    f" Instructions will be:\n {selected_instr}"
                )

            screen_message.render(message)

            if tm or tc or skipped or passed:
                screen_message.clear()
                # NEW: kill haptics immediately so the wheel is quiet during the wait
                try:
                    if hasattr(env, "agent") and hasattr(env.agent, "policy"):
                        ctrl = getattr(env.agent.policy, "controller", None)
                        if hasattr(ctrl, "pause_autocenter"):
                            ctrl.pause_autocenter()
                except Exception as e:
                    print(f"⚠️ could not pause wheel after scenario: {e}")
                break







        # Save manual inputs + which instruction was used
        if global_config["agent_policy"] == HybridEgoCarPolicy and not skipped:
            traj_info = {
                "positions": positions,
                "headings": headings,
                "velocities": velocities,
                "accelerations": accelerations,
                "steering_signals": steering_signals,
                # Metadata about the randomly chosen instruction/onset
                "selected_instruction": selected_instr,
                "selected_onset_manual": int(selected_onset),
                "selected_instruction_index": int(selected_idx),
                "controller": controller,
            }
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(traj_info, f, indent=1)
    env.close()


if __name__ == "__main__":
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description="Metadrive Hybrid Data Collection")
    parser.add_argument(
        "-c",
        "--config_path",
        default="configs",
        type=str,
        help="Path to the config files for the scenarios",
    )
    parser.add_argument(
        "-d", "--data_path", default=waymo_data, type=str, help="Path to the data files for the scenarios"
    )
    parser.add_argument("--replay", action=argparse.BooleanOptionalAction)
    parser.add_argument("--controller", type=str, default="keyboard",help="Input device to use: e.g., 'keyboard', 'xbox', 'wheel', 'enhanced_wheel'"
    )

    args = parser.parse_args()
    try:
        configs = [
            os.path.join(args.config_path, f)
            for f in os.listdir(args.config_path)
            if f.endswith(".json")
        ]
    except Exception:
        configs = None
        print("Config folder does not exist")
    if configs and not args.replay:
        collect(configs, args.data_path, controller=args.controller)
    elif configs and args.replay:
        replay(configs, args.data_path, controller=args.controller)


'''
    try:
        configs = [os.path.join(args.config_path, config) for config in os.listdir(args.config_path)]
    except Exception:
        configs = None
        print("Config folder does not exist")
'''




