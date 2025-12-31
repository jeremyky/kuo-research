#!/usr/bin/env python
# Simplified data collection pipeline (requires configs present):
#   PRE_START_GATE (paused + 7→0 HUD + push-to-center, then clamp ON)
#   → REPLAY_LOCKED (ghost; wheel locked & clamped; HUD takeover countdown)
#   → UNLOCK_PREPARE (~1s before takeover; inputs unlocked, clamp OFF, spring stays ON but force=0)
#   → MANUAL (user drives; spring ON with speed-based force & center hysteresis+damping; safety unlock)
#   → SAVE → NEXT

# Fix OpenBLAS thread explosion (must be BEFORE numpy import!)
import os as _os
# Auto-detect optimal thread count based on CPU cores
_cpu_count = _os.cpu_count() or 4
_optimal_threads = min(8, max(4, _cpu_count // 2))  # Use half cores, capped at 4-8 threads
_thread_str = str(_optimal_threads)
_os.environ['OPENBLAS_NUM_THREADS'] = _thread_str
_os.environ['MKL_NUM_THREADS'] = _thread_str
_os.environ['NUMEXPR_NUM_THREADS'] = _thread_str
_os.environ['OMP_NUM_THREADS'] = _thread_str
print(f"🧵 Detected {_cpu_count} CPU cores → using {_optimal_threads} threads for numpy operations")

import argparse
import json
import os
import random
import math
import time
import traceback

# NumPy 2.0+ compatibility fix for pickle files created with older numpy
import sys
import numpy as np
if not hasattr(np, '_core'):
    np._core = np.core
    sys.modules['numpy._core'] = np.core
if not hasattr(np, '_core._multiarray_umath'):
    try:
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
    except AttributeError:
        pass

from metadrive.engine.engine_utils import get_global_config, get_engine
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.custom_scenario_env import CustomScenarioEnv
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.scenario.utils import read_dataset_summary
from gymnasium.spaces import Discrete
from PIL import Image

# Import enhanced controller if available
try:
    from enhanced_steering_wheel_controller import EnhancedSteeringWheelController
except Exception:
    EnhancedSteeringWheelController = None

# Fallback controller creation if policy/controller missing
try:
    from metadrive.policy.manual_control_policy import get_controller
except Exception:
    get_controller = None

from partial_manual_policy import HybridEgoCarPolicy, HybridEgoCarReplayPolicy

# ---------- dataset path ----------
waymo_data = AssetLoader.file_path(AssetLoader.asset_path, "waymo", unix_style=False)

# ---------- constants ----------
MAX_STEPS = 100000
HybridEgoCarPolicy.set_extra_input_space(Discrete(MAX_STEPS))
HybridEgoCarReplayPolicy.set_extra_input_space(Discrete(MAX_STEPS))

# Timings
PRE_GATE_SECONDS = 7
UNLOCK_EARLY_SECONDS = 1.0   # seconds before onset to allow pre-position

# ---------- TUNING: FAST RETURN + LOW-SPEED=2 km/h (displacement-driven) ----------

DEFAULT_GHOST_CLAMP_FORCE     = 100


# Strong, quick spring (keeps displacement as primary driver)
DEFAULT_MANUAL_FORCE_BASE     = 1250
DEFAULT_MANUAL_FORCE_MIN      = 1100
DEFAULT_MANUAL_FORCE_MAX      = 3200
DEFAULT_MANUAL_SPEED_KMH_MAX  = 120.0    # higher → weaker speed dependence overall
DEFAULT_MANUAL_FORCE_SHAPE_P  = 1.20     # >1 → gentle speed scaling


# Deadzone & ramp (engage early, ramp fast with displacement)
DEFAULT_MANUAL_DEAD_IN_DEG      = 0.12
DEFAULT_MANUAL_DEAD_OUT_DEG     = 0.20
DEFAULT_MANUAL_SOFT_OUT_DEG     = 24.0    # lower → quicker buildup vs angle
DEFAULT_MANUAL_CENTER_EXPONENT  = 0.70    # <1 → more force just off center (faster recenter)


# Return-speed controls (snappy but stable)
DEFAULT_MANUAL_DAMP_K        = 32.0       # lower → less braking → faster return
DEFAULT_MANUAL_DAMP_CAP      = 18.0
DEFAULT_MANUAL_FORCE_ALPHA   = 0.96       # higher → faster torque updates


# Low-speed assist (still pulls at crawl, just slower)
DEFAULT_MANUAL_LOWSPD_KMH    = 2.0        # requested
DEFAULT_MANUAL_LOWSPD_GAIN   = 0.72       # keeps pullback at 0–2 km/h without feeling glued


# Mild high-speed heaviness (displacement does most of the work)
DEFAULT_SPEED_RESISTANCE_MIN      = 0
DEFAULT_SPEED_RESISTANCE_MAX      = 280
DEFAULT_SPEED_RESISTANCE_THRESHOLD = 34.0
DEFAULT_SPEED_RESISTANCE_SHAPE    = 1.6






























# ---------- helpers ----------
def clear_leaked_autocenter_config():
    """
    Call this at the start of other driving scripts to clear any leaked
    autocenter config from data_collection.py runs.
    
    Usage in other scripts:
        from data_collection import clear_leaked_autocenter_config
        clear_leaked_autocenter_config()
    """
    try:
        gc = get_global_config()
        cleanup_keys = [
            "autocenter_locked", "force_controller_init", "takeover",
            "manual_force_min", "manual_force_max", "manual_speed_kmh_max",
            "manual_force_shape_p", "manual_dead_in_deg", "manual_dead_out_deg",
            "manual_soft_out_deg", "manual_center_exponent", "manual_damping_k",
            "manual_damping_cap", "manual_low_speed_hold_kmh", "manual_low_speed_gain",
            "manual_force_smooth_alpha", "speed_resistance_min", "speed_resistance_max",
            "speed_resistance_threshold", "speed_resistance_shape", "ghost_clamp_force"
        ]
        cleared = []
        for key in cleanup_keys:
            if key in gc:
                try:
                    del gc[key]
                    cleared.append(key)
                except Exception:
                    pass
        if cleared:
            print(f"[CLEANUP] Cleared leaked autocenter config keys: {', '.join(cleared)}")
        else:
            print("[CLEANUP] No leaked autocenter config found (clean start)")
    except Exception as e:
        print(f"[CLEANUP] Error clearing leaked config: {e}")

def _load_configs(config_paths):
    out = {}
    if not config_paths:
        return out
    for c in config_paths:
        if not c.endswith(".json"):
            continue
        key = os.path.splitext(os.path.basename(c))[0]
        with open(c, "r") as f:
            out[key] = json.load(f)
    return out

def _pick_instruction_and_onset(cfg):
    instr_list = cfg.get("instructions")
    onsets = cfg.get("onset_manuals")
    if isinstance(instr_list, list) and instr_list:
        idx = random.randrange(len(instr_list))
        instr = str(instr_list[idx])
        onset = int(onsets[idx]) if isinstance(onsets, list) and idx < len(onsets) else int(cfg.get("onset_manual", 0))
        return instr, onset, idx
    instr = str(cfg.get("instructions") or cfg.get("instructions_legacy") or cfg.get("selected_instruction") or "follow the lane")
    onset = int(cfg.get("onset_manual", 0) or cfg.get("selected_onset_manual", 0))
    return instr, onset, 0

def _controller_if_any(env):
    try:
        policy = getattr(getattr(env, "agent", None), "policy", None)
        return getattr(policy, "controller", None), policy
    except Exception:
        return None, None

def _diagnose_controllers(env, label):
    eng = get_engine()
    def _id(x): return None if x is None else id(x)
    print(f"\n=== DEBUG ({label}) controller scan ===")
    try: print(f"  engine.controller      -> {type(getattr(eng,'controller',None)).__name__ if getattr(eng,'controller',None) else None:>28} id={_id(getattr(eng,'controller',None))}")
    except Exception: print("  engine.controller      -> <error>")
    try: print(f"  engine._controller     -> {type(getattr(eng,'_controller',None)).__name__ if getattr(eng,'_controller',None) else None:>28} id={_id(getattr(eng,'_controller',None))}")
    except Exception: print("  engine._controller     -> <error>")
    try: print(f"  env.controller         -> {type(getattr(env,'controller',None)).__name__ if getattr(env,'controller',None) else None:>28} id={_id(getattr(env,'controller',None))}")
    except Exception: print("  env.controller         -> <error>")
    try: print(f"  env.engine.controller  -> {type(getattr(getattr(env,'engine',None),'controller',None)).__name__ if getattr(getattr(env,'engine',None),'controller',None) else None:>28} id={_id(getattr(getattr(env,'engine',None),'controller',None))}")
    except Exception: print("  env.engine.controller  -> <error>")
    try:
        pol = getattr(getattr(env, "agent", None), "policy", None)
        print(f"  policy                 -> {type(pol).__name__ if pol else None:>28} id={_id(pol)}")
        print(f"  policy.controller      -> {type(getattr(pol,'controller',None)).__name__ if getattr(pol,'controller',None) else None:>28} id={_id(getattr(pol,'controller',None))}")
    except Exception:
        print("  policy/controller scan -> <error>")
    print("=== END DEBUG ===\n")

def _ensure_controller_loaded(env):
    ctrl, policy = _controller_if_any(env)
    if ctrl is not None:
        if hasattr(policy, "ensure_controller_loaded"):
            try: policy.ensure_controller_loaded()
            except Exception: pass
        return True
    if policy is not None:
        created = False
        for method in ("create_controller", "_init_controller", "ensure_controller_loaded"):
            if hasattr(policy, method):
                try:
                    getattr(policy, method)()
                    created = getattr(policy, "controller", None) is not None
                    if created: break
                except Exception:
                    pass
        if not created:
            get_global_config().update({"force_controller_init": True})
        return created
    print("[CTRL] No policy to attach a controller to.")
    return False

def _unify_controller_handles(env):
    eng = get_engine()
    eng_ctrl = getattr(eng, "controller", None)
    pol_ctrl, policy = _controller_if_any(env)
    print(f"[CTRL] Unify start: eng_ctrl={type(eng_ctrl).__name__ if eng_ctrl else None} id={id(eng_ctrl) if eng_ctrl else None}  "
          f"pol_ctrl={type(pol_ctrl).__name__ if pol_ctrl else None} id={id(pol_ctrl) if pol_ctrl else None}")

    chosen = eng_ctrl or pol_ctrl
    if chosen is None and get_controller is not None:
        print("[CTRL] Neither engine nor policy has a controller. Trying policy ensure path …")
        _ensure_controller_loaded(env)
        pol_ctrl, _ = _controller_if_any(env)
        chosen = pol_ctrl
        if chosen is None:
            print("[CTRL] Fallback: get_controller('enhanced_wheel') …")
            try:
                chosen = get_controller(
                    get_global_config().get("controller", "enhanced_wheel"), 
                    pygame_control=False,
                    enable_autocenter=True
                )
                setattr(eng, "controller", chosen)
                print("[CTRL] Mounted fallback controller on engine.controller")
            except Exception as e:
                print(f"[CTRL] Fallback controller creation failed: {e}")
                chosen = None

    print(f"[CTRL] Unify done: chosen={type(chosen).__name__ if chosen else None} id={id(chosen) if chosen else None}")
    return chosen

def _close_wheel_after_scenario(env, restore_force=None):
    """
    Close wheel and restore state after scenario.
    
    Args:
        env: The environment
        restore_force: If provided, restore autocenter_force to this value before cleanup
    """
    try:
        policy = getattr(getattr(env, "agent", None), "policy", None)
        ctrl = getattr(policy, "controller", None)
        
        # Restore original force if provided
        if ctrl and restore_force is not None and hasattr(ctrl, "autocenter_force"):
            try:
                ctrl.autocenter_force = int(restore_force)
                print(f"[CLEANUP] Restored autocenter_force to {restore_force}")
            except Exception as e:
                print(f"[CLEANUP] Failed to restore force: {e}")
        
        # Cleanup controller
        if ctrl and hasattr(ctrl, "cleanup"):
            ctrl.cleanup()
        
        # Clear engine controller reference
        eng = get_engine()
        if hasattr(eng, "controller"):
            setattr(eng, "controller", None)
            
        # Clear any lingering global config values that might leak
        gc = get_global_config()
        cleanup_keys = [
            "autocenter_locked", "force_controller_init", "takeover",
            "manual_force_min", "manual_force_max", "manual_speed_kmh_max",
            "manual_force_shape_p", "manual_dead_in_deg", "manual_dead_out_deg",
            "manual_soft_out_deg", "manual_center_exponent", "manual_damping_k",
            "manual_damping_cap", "manual_low_speed_hold_kmh", "manual_low_speed_gain",
            "manual_force_smooth_alpha", "speed_resistance_min", "speed_resistance_max",
            "speed_resistance_threshold", "speed_resistance_shape", "ghost_clamp_force"
        ]
        for key in cleanup_keys:
            if key in gc:
                try:
                    del gc[key]
                except Exception:
                    pass
        print("[CLEANUP] Cleared global config autocenter values")
        
    except Exception as e:
        print(f"⚠️ could not close wheel after scenario: {e}")

def _lock_wheel(ctrl, spring=True):
    if not ctrl: return
    try:
        if hasattr(ctrl, "set_spring") and spring: ctrl.set_spring(True)
        if hasattr(ctrl, "lock_input"):          ctrl.lock_input(True)
        print("[PHASE] LOCK (spring ON, inputs locked)")
    except Exception as e:
        print(f"[PHASE] lock error: {e}")

def _unlock_inputs_only(ctrl):
    """Do NOT stop spring; just allow inputs through."""
    if not ctrl: return
    try:
        if hasattr(ctrl, "lock_input"): ctrl.lock_input(False)
        print("[PHASE] UNLOCK INPUTS (spring untouched)")
    except Exception as e:
        print(f"[PHASE] unlock inputs error: {e}")

def _pre_start_gate(engine, ctrl, instruction, seconds=PRE_GATE_SECONDS, deadband_deg=2.0, settle_s=0.4):
    gc = get_global_config()
    screen = ScreenMessage()
    screen.POS = (-0.25, 0.75)

    print("[PHASE] === PRE_START_GATE begin ===")
    if ctrl and hasattr(ctrl, "hard_reset_sync"):
        print("[PHASE] hard_reset_sync()")
        try:
            ctrl.hard_reset_sync(
                device_reset_time_s=0.0,
                center_lock_s=float(seconds) + 0.5,
                spring_boost=1.15  # gentle boost (avoid scary snap)
            )
        except Exception as e:
            print(f"[PHASE] hard_reset_sync error: {e}")

    try:
        if ctrl and hasattr(ctrl, "set_clamp"):
            ctrl.set_clamp(False)   # PUSH mode to center
            print("[PHASE] set_clamp(False) -> PUSH centering")
    except Exception as e:
        print(f"[PHASE] set_clamp(False) error: {e}")

    # Warm-up loop to ensure haptics thread is alive
    warm_deadline = time.monotonic() + 0.30
    while time.monotonic() < warm_deadline:
        if ctrl and hasattr(ctrl, "process_input"):
            try: ctrl.process_input(None)
            except Exception: pass
        try: engine.render_frame()
        except Exception: pass
        if hasattr(engine, "taskMgr"):
            try: engine.taskMgr.step()
            except Exception: pass
        time.sleep(0.01)

    # Optional auto-cal
    if ctrl and hasattr(ctrl, "auto_calibrate_sign"):
        print("[PHASE] auto_calibrate_sign()")
        try: ctrl.auto_calibrate_sign(duration_s=0.25, poll_hz=100)
        except Exception: pass

    _lock_wheel(ctrl, spring=True)
    gc.update({"autocenter_locked": True})

    # Gentle temporary boost for centering (lower and capped)
    orig_force = None
    try:
        if ctrl and hasattr(ctrl, "autocenter_force"):
            orig_force = int(ctrl.autocenter_force)
            gc_local = get_global_config()
            boost_mult = float(gc_local.get("pre_gate_force_mult", 1.10))
            boost_cap  = int(gc_local.get("pre_gate_force_cap", 4200))
            min_force  = int(gc_local.get("pre_gate_force_min", 1200))
            new_force  = max(min_force, min(boost_cap, int(orig_force * boost_mult)))
            ctrl.autocenter_force = new_force
            print(f"[PHASE] temporary force boost: {orig_force} -> {ctrl.autocenter_force} "
                  f"(mult={boost_mult}, cap={boost_cap}, min={min_force})")
    except Exception as e:
        print(f"[PHASE] force boost error: {e}")

    # Countdown
    t_end = time.monotonic() + float(seconds)
    while True:
        now = time.monotonic()
        left = max(0, int(round(t_end - now)))

        if ctrl and hasattr(ctrl, "process_input"):
            try: ctrl.process_input(None)
            except Exception: pass

        screen.render(
            "preparing waymo scenario...\n"
            f"Instruction:\n {instruction}\n"
            "wheel is locked & auto-centering strongly\n"
            f"Starting in: {left}s\n"
        )
        try: engine.render_frame()
        except Exception: pass
        if hasattr(engine, "taskMgr"):
            try: engine.taskMgr.step()
            except Exception: pass

        if now >= t_end: break
        time.sleep(0.05)

    # Dwell wait (grace)
    ready_deadline = time.monotonic() + 3.0
    while True:
        ok = True
        if ctrl and hasattr(ctrl, "is_centered"):
            try: ok = bool(ctrl.is_centered(deadband_deg, settle_s))
            except Exception: ok = True
        if ok: break

        if ctrl and hasattr(ctrl, "process_input"):
            try: ctrl.process_input(None)
            except Exception: pass
        screen.render(
            "preparing waymo scenario...\n"
            f"Instruction:\n {instruction}\n"
            "waiting for wheel to center...\n"
        )
        try: engine.render_frame()
        except Exception: pass
        if hasattr(engine, "taskMgr"):
            try: engine.taskMgr.step()
            except Exception: pass

        if time.monotonic() >= ready_deadline:
            print("⚠️ pre-start gate: wheel not centered in grace window; continuing anyway.")
            break
        time.sleep(0.05)

    # Enter CLAMP for ghost
    try:
        if ctrl and hasattr(ctrl, "set_clamp"):
            ctrl.set_clamp(True)
            print("[PHASE] set_clamp(True) -> CLAMP for ghost")
    except Exception as e:
        print(f"[PHASE] set_clamp(True) error: {e}")

    screen.clear()

    # restore pre-gate force
    try:
        if ctrl and (orig_force is not None) and hasattr(ctrl, "autocenter_force"):
            ctrl.autocenter_force = orig_force
            print(f"[PHASE] restore force: {orig_force}")
    except Exception:
        pass

# ---------- REPLAY (compat) ----------
def replay(configs, data, controller="keyboard", inputs_dir="manual_inputs", only_seed=None, only_sid=None, onset_lead=7, use_topdown=False, skip_ghost=False):
    configs = _load_configs(configs)
    dataset_summary = read_dataset_summary(data)
    num_scenarios = len(dataset_summary[1])
    
    if skip_ghost:
        print("\n" + "="*60)
        print("⏩ SKIP GHOST MODE: Playing only human inputs")
        print("="*60)
        print("Original Waymo trajectory will be skipped")
        print("Replay starts from takeover point")
        print("="*60 + "\n")
    
    if use_topdown:
        print("\n" + "="*60)
        print("🎥 TOPDOWN RENDERING MODE ENABLED")
        print("="*60)
        print("Window: 1200x1200 pixels")
        print("Scaling: 5 pixels/meter")
        print("Trail length: 60 frames")
        print("Semantic map: ON (color-coded vehicles)")
        print("Full trajectory: ON (shows complete path)")
        print("Camera: Follow ego vehicle")
        print("="*60 + "\n")

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
            # Smooth 60 FPS rendering (doesn't change physics/simulation speed)
            "force_render_fps": 60,  # Smooth 60 FPS visuals
            # Larger window for better visibility (default is 1200x900)
            "window_size": (1920, 1080),  # Full HD resolution
            # Better camera view - see more surroundings
            "camera_dist": 10.0,  # Further back (default 7.5)
            "camera_height": 3.5,  # Higher up (default 2.2)
            "camera_fov": 80,  # Wider view (default 65)
        }
    )
    
    seeds = [only_seed] if (only_seed is not None) else range(num_scenarios)

    for seed in seeds:
        if data != waymo_data:
            sid_hint = dataset_summary[1][seed].split("_")[3].split(".")[0]
            if sid_hint not in configs:
                print(f"[Replay] Seed {seed} sid_hint={sid_hint} not in configs, skipping.")
                continue
            
        o, _ = env.reset(seed=seed)
        engine = get_engine()
        sid = engine.data_manager.current_scenario["id"]
        
        # Filter by scenario ID if specified
        if only_sid is not None and sid != only_sid:
            print(f"[Replay] Seed {seed} sid={sid} != only_sid={only_sid}, skipping.")
            continue
        
        if sid not in configs:
            print(f"[Replay] Seed {seed} sid={sid} not in configs, skipping.")
            continue
        cfg = configs[sid]

        input_path = os.path.join(inputs_dir, f"{sid}.json")
        selected_onset, selected_instr, final_index = None, None, MAX_STEPS
        try:
            with open(input_path, "r") as fh:
                mi = json.load(fh)
            selected_onset = int(mi.get("selected_onset_manual")) if "selected_onset_manual" in mi else None
            selected_instr = mi.get("selected_instruction")
            final_index = int(mi.get("final_index", MAX_STEPS))
            print(f"[Replay] Using inputs: {input_path} (onset={selected_onset}, final_index={final_index})")
        except FileNotFoundError:
            print(f"[Replay] No inputs for sid={sid} at {input_path}. Falling back to config. (Skipping input replay).")
        except Exception as e:
            print(f"[Replay] Failed to read inputs for sid={sid}: {e}")

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

        # Determine start index based on skip_ghost flag
        if skip_ghost:
            # Start from takeover point (skip ghost phase)
            try:
                with open(input_path, "r") as fh:
                    mi = json.load(fh)
                start_index = int(mi.get("takeover_start_index", selected_onset))
                if start_index is None:
                    start_index = selected_onset
                print(f"[Replay] Skipping ghost phase, starting from takeover at index {start_index}")
            except Exception:
                start_index = selected_onset
        else:
            # Normal replay: start from beginning
            start_index = 0
        
        # CRITICAL: Set replay_control to False initially - the policy will set it to True when it loads inputs!
        # This matches the working reference implementation
        get_global_config().update(
            {
                "onset_manual": max(selected_onset - int(onset_lead), 0),
                "manual_inputs": input_path,
                "replay_control": False,  # Policy sets this to True after loading inputs
                "allowed_more_steps": cfg.get("extra_steps", 1),
                "reactive_traffic": False,
                "trajectory_dots": False,
                "replay_use_states": False,  # Must be False to replay actions (not states)
            }
        )

        screen_message = ScreenMessage()
        screen_message.POS = (-0.25, 0.75)

        max_idx = min(MAX_STEPS, int(final_index), MAX_STEPS)
        # Always loop from 0 - the policy handles what to do at each index
        user_quit = False
        for i in range(max_idx):
            # Check for window close or escape key to allow early exit
            try:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n[Replay] Window closed by user - exiting...")
                        user_quit = True
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                            print("\n[Replay] Escape/Q pressed - exiting...")
                            user_quit = True
                            break
                        elif event.key == pygame.K_z:
                            print("\n[Replay] Z pressed - skipping to next scenario...")
                            user_quit = True
                            break
            except Exception:
                pass  # pygame not available or error checking events
            
            if user_quit:
                break
            
            o, r, tm, tc, info = env.step({"action": [0, 0], "extra": i})

            # Topdown rendering if requested
            if use_topdown:
                try:
                    env.render(
                        mode="topdown",
                        window=True,
                        screen_size=(1200, 1200),
                        scaling=5,
                        film_size=(1500, 1500),  # Reduced from 2000x2000 for better FPS
                        num_stack=30,  # Reduced from 60 for faster rendering
                        history_smooth=0,
                        draw_target_vehicle_trajectory=True,
                        semantic_map=True,
                        target_agent_heading_up=True,
                        screen_record=False,
                    )
                except Exception as e:
                    print(f"[Replay] Topdown render error: {e}")
            
            # Regular message rendering (only if not using topdown, to avoid conflicts)
            if not use_topdown:
                msg = (
                    f"Replaying Human Inputs\nInstruction: {selected_instr}"
                    if get_global_config().get("replay_control", False)
                    else "Replaying Original Trajectory"
                )
                screen_message.render(msg)
            
            if tm or tc or user_quit:
                if not use_topdown:
                    screen_message.clear()
                _close_wheel_after_scenario(env)
                break
        
        if not use_topdown:
            screen_message.clear()

    env.close()
    return

# ---------- small state holder for MANUAL hysteresis/damping ----------
class _manual_center_state:
    pass

# ---------- bad scenario tracking ----------
def _load_bad_scenarios(bad_scenarios_path="bad_scenarios.json"):
    """Load the list of scenarios marked for deletion."""
    if not os.path.exists(bad_scenarios_path):
        return []
    try:
        with open(bad_scenarios_path, "r") as f:
            data = json.load(f)
        return data.get("scenarios", [])
    except Exception as e:
        print(f"⚠️ Failed to load bad scenarios: {e}")
        return []

def _save_bad_scenario(sid, seed, instruction, reason, bad_scenarios_path="bad_scenarios.json"):
    """Mark a scenario as bad and save to reviewable JSON file."""
    bad_list = _load_bad_scenarios(bad_scenarios_path)
    
    # Check if already marked
    if any(item["sid"] == sid for item in bad_list):
        print(f"⚠️ Scenario {sid} already marked as bad")
        return
    
    # Add to list with metadata
    bad_list.append({
        "sid": sid,
        "seed": seed,
        "instruction": instruction,
        "reason": reason,
        "marked_at": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Save to file
    try:
        with open(bad_scenarios_path, "w") as f:
            json.dump({
                "scenarios": bad_list,
                "count": len(bad_list),
                "note": "Review these scenarios and delete their config files if needed"
            }, f, indent=2)
        print(f"\n{'='*60}")
        print(f"🗑️  MARKED AS BAD: {sid}")
        print(f"{'='*60}")
        print(f"Reason: {reason}")
        print(f"Total bad scenarios: {len(bad_list)}")
        print(f"Review list: {bad_scenarios_path}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"❌ Failed to save bad scenario: {e}")

# ---------- COLLECT ----------
def collect(
    configs, data, controller="keyboard",
    out_dir="manual_inputs", samples_dir="samples",
    only_seed=None, only_sid=None, onset_lead=7
):
    """
    Pipeline:
      - PRE_START_GATE: 7→0 HUD; hard reset wheel; PUSH to center; wait until centered; enter CLAMP.
      - REPLAY_LOCKED: ghost; wheel locked + clamped until ~1s before onset.
      - UNLOCK_PREPARE: ~1s before onset -> inputs unlocked, clamp OFF, spring stays ON (force=0) so it's free.
      - MANUAL: on first takeover frame -> spring ON + clamp OFF; speed-based force with center hysteresis+damping; record.
    
    Keyboard shortcuts during collection:
      - Z: Finished early (save and continue)
      - X: Instructions impossible (save and continue)
      - D: Mark scenario as bad/to-delete (saves to bad_scenarios.json for review)
      - Q/ESC: Quit (save progress)
    """
    configs = _load_configs(configs)
    dataset_summary = read_dataset_summary(data)
    num_scenarios = len(dataset_summary[1])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
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
            # Larger window for better visibility during data collection
            "window_size": (1920, 1080),  # Full HD resolution
            # Better camera view - see more surroundings while driving
            "camera_dist": 10.0,  # Further back (default 7.5)
            "camera_height": 3.5,  # Higher up (default 2.2)
            "camera_fov": 80,  # Wider view (default 65)
        }
    )

    engine = get_engine()
    seeds = [only_seed] if (only_seed is not None) else range(num_scenarios)
    
    # Track scenario progress
    total_configs = len(configs)
    scenarios_completed = 0

    for seed in seeds:
        if data != waymo_data:
            sid_hint = dataset_summary[1][seed].split("_")[3].split(".")[0]
            if sid_hint not in configs:
                print(f"[Collect] Seed {seed} sid_hint={sid_hint} not in configs, skipping.")
                continue

        get_global_config().update({"force_controller_init": True})
        
        try:
            o, _ = env.reset(seed=seed)
        except Exception:
            print("[Collect] reset failed; skipping scenario")
            print(traceback.format_exc())
            continue

        engine = get_engine()

        print("\n[CTRL] After env.reset()")
        _diagnose_controllers(env, "post-reset (before ensure)")

        _ensure_controller_loaded(env)

        print("\n[CTRL] After ensure_controller_loaded()")
        _diagnose_controllers(env, "after ensure")

        ctrl_cached = _unify_controller_handles(env)

        print("\n[CTRL] After unify")
        _diagnose_controllers(env, "after unify")

        if ctrl_cached is None:
            print("❌ No controller found before pre-start gate — physical centering will NOT run.")
        else:
            print("✅ Controller ready before pre-start gate.")

        gc = get_global_config()
        sid = engine.data_manager.current_scenario["id"]
        
        # Filter by scenario ID if specified
        if only_sid is not None and sid != only_sid:
            print(f"[Collect] Seed {seed} sid={sid} != only_sid={only_sid}, skipping.")
            _close_wheel_after_scenario(env, restore_force=None)
            continue
        
        if sid not in configs:
            print(f"[Collect] Seed {seed} sid={sid} not in configs, skipping.")
            _close_wheel_after_scenario(env, restore_force=None)
            continue
        
        cfg = configs[sid]
        instr, onset_manual, instr_idx = _pick_instruction_and_onset(cfg)
        output_path = os.path.join(out_dir, f"{sid}.json")
        
        # Calculate scenario position
        scenarios_remaining = total_configs - scenarios_completed
        current_scenario_num = scenarios_completed + 1
        
        print("\n" + "="*80)
        print(f"🎬 SCENARIO {current_scenario_num}/{total_configs} ({scenarios_remaining} remaining)")
        print(f"📋 ID: {sid}")
        print("="*80)
        print(f"📍 Onset frame: {onset_manual}")
        print(f"📝 Instruction: '{instr}'")
        print(f"💾 Will save to: {output_path}")
        print(f"🎮 Controller: {controller}")
        print("="*80 + "\n")

        # PRE_START_GATE
        _pre_start_gate(engine, ctrl_cached, instr, seconds=int(PRE_GATE_SECONDS))

        # Policy-level flags
        gc.update({
            "onset_manual": max(int(onset_manual) - int(onset_lead), 0),
                "allowed_more_steps": cfg.get("extra_steps", 1),
                "reactive_traffic": cfg.get("reactive_traffic", False),
                "takeover": False,
            "trajectory_dots": False,
            "autocenter_locked": True,
            "replay_use_states": True,
            "replay_control": False,
        })

        # HUD
        screen = ScreenMessage()
        screen.POS = (-0.25, 0.75)

        # Recorder buffers
        positions, velocities = [], []
        headings, heading_theta = [], []
        steering_signals, accelerations = [], []
        is_manual_flags, frames = [], []
        timestamps = []

        takeover_start_index = None
        collect_start_index = 0
        prev_takeover_flag = False
        sampled_png = False
        output_path = os.path.join(out_dir, f"{sid}.json")
        t0 = engine.get_global_time() if hasattr(engine, "get_global_time") else 0.0

        # FPS
        try:
            fps = int(getattr(engine, "global_config", {}).get("target_FPS", 0))
        except Exception:
            fps = 0
        if fps <= 0:
            fps = 20

        show_window_frames   = int(3.0 * fps)
        unlock_early_frames  = int(float(UNLOCK_EARLY_SECONDS) * fps)

        # Phase-local force settings - capture original force for restoration later
        force_orig  = getattr(ctrl_cached, "autocenter_force", None)
        if force_orig is not None:
            print(f"[FORCE] Captured original autocenter_force: {force_orig}")
        else:
            print("[FORCE] No original autocenter_force found on controller")
        ghost_force = int(get_global_config().get("ghost_clamp_force",  DEFAULT_GHOST_CLAMP_FORCE))

        # Manual force dynamic state (speed curve)
        manual_min   = float(get_global_config().get("manual_force_min",     DEFAULT_MANUAL_FORCE_MIN))
        manual_max   = float(get_global_config().get("manual_force_max",     DEFAULT_MANUAL_FORCE_MAX))
        speed_max    = float(get_global_config().get("manual_speed_kmh_max", DEFAULT_MANUAL_SPEED_KMH_MAX))
        shape_p      = float(get_global_config().get("manual_force_shape_p", DEFAULT_MANUAL_FORCE_SHAPE_P))

        # Manual hysteresis/damping knobs
        dead_in_deg     = float(get_global_config().get("manual_dead_in_deg",      DEFAULT_MANUAL_DEAD_IN_DEG))
        dead_out_deg    = float(get_global_config().get("manual_dead_out_deg",     DEFAULT_MANUAL_DEAD_OUT_DEG))
        soft_out_deg    = float(get_global_config().get("manual_soft_out_deg",     DEFAULT_MANUAL_SOFT_OUT_DEG))
        center_exponent = float(get_global_config().get("manual_center_exponent",  DEFAULT_MANUAL_CENTER_EXPONENT))
        kd_damp         = float(get_global_config().get("manual_damping_k",        DEFAULT_MANUAL_DAMP_K))
        damp_cap        = float(get_global_config().get("manual_damping_cap",      DEFAULT_MANUAL_DAMP_CAP))
        low_spd_kmh     = float(get_global_config().get("manual_low_speed_hold_kmh", DEFAULT_MANUAL_LOWSPD_KMH))
        low_spd_gain    = float(get_global_config().get("manual_low_speed_gain",   DEFAULT_MANUAL_LOWSPD_GAIN))
        alpha           = float(get_global_config().get("manual_force_smooth_alpha", DEFAULT_MANUAL_FORCE_ALPHA))
        
        # Speed-sensitive steering resistance
        resist_min      = float(get_global_config().get("speed_resistance_min",    DEFAULT_SPEED_RESISTANCE_MIN))
        resist_max      = float(get_global_config().get("speed_resistance_max",    DEFAULT_SPEED_RESISTANCE_MAX))
        resist_thresh   = float(get_global_config().get("speed_resistance_threshold", DEFAULT_SPEED_RESISTANCE_THRESHOLD))
        resist_shape    = float(get_global_config().get("speed_resistance_shape",  DEFAULT_SPEED_RESISTANCE_SHAPE))

        manual_force_curr = 0.0  # EMA state
        last_phase = None  # 'ghost', 'prep', 'manual'
        term_reason = "unknown"
        bad_scenario = False
        user_quit = False  # Track if user pressed escape/closed window
        marked_bad = False  # Track if user marked scenario as bad (D key)

        # init hysteresis state
        _manual_center_state.inside = True
        _manual_center_state.prev_err = 0.0
        _manual_center_state.prev_t = time.monotonic()
        _manual_center_state.ema = 0.0

        for i in range(MAX_STEPS):
            # Check for window close or escape key
            try:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n[Collect] Window closed by user - saving progress...")
                        term_reason = "user_quit"
                        user_quit = True
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                            print("\n[Collect] Escape/Q pressed - saving progress...")
                            term_reason = "user_quit"
                            user_quit = True
                            break
                        elif event.key == pygame.K_z:
                            print("\n[Collect] Z pressed - finished early, saving progress...")
                            term_reason = "finished_early"
                            user_quit = True
                            break
                        elif event.key == pygame.K_x:
                            print("\n[Collect] X pressed - instructions impossible, saving progress...")
                            term_reason = "instructions_impossible"
                            user_quit = True
                            break
                        elif event.key == pygame.K_d:
                            print("\n[Collect] D pressed - marking scenario as BAD...")
                            marked_bad = True
                            _save_bad_scenario(sid, seed, instr, "marked_during_collection")
                            # Don't quit - continue collecting but mark for review
            except Exception:
                pass  # pygame not available or error checking events
            
            if user_quit:
                break
            
            try:
                o, r, tm, tc, info = env.step({"action": [0, 0], "extra": i})
            except Exception:
                screen.clear()
                print("[Collect] Error stepping scenario; skipping to next")
                print(traceback.format_exc())
                term_reason = "scenario_error"
                bad_scenario = True
                break

            takeover_flag   = bool(get_global_config().get("takeover", False))
            onset           = int(get_global_config().get("onset_manual", 0))
            frames_to_onset = onset - i

            in_ghost = (not takeover_flag and frames_to_onset > unlock_early_frames)
            in_prep  = (not takeover_flag and 0 < frames_to_onset <= unlock_early_frames)
            in_manual= (takeover_flag)

            phase = "ghost" if in_ghost else ("prep" if in_prep else ("manual" if in_manual else "post"))
            
            # ENFORCE lock state EVERY frame during ghost (not just on phase entry)
            # This prevents the wheel from becoming unlocked if controller state drifts
            if phase == "ghost" and ctrl_cached:
                if hasattr(ctrl_cached, "lock_input"):
                    try: ctrl_cached.lock_input(True)
                    except Exception: pass
                if hasattr(ctrl_cached, "set_clamp"):
                    try: ctrl_cached.set_clamp(True)
                    except Exception: pass

            # feed haptics each frame (AFTER enforcing lock state)
            if ctrl_cached and hasattr(ctrl_cached, "process_input"):
                try: ctrl_cached.process_input(env.agent)
                except Exception: pass

            if phase != last_phase:
                print(f"[PHASE] -> {phase.upper()} (i={i}, frames_to_onset={frames_to_onset})")
                if phase == "ghost":
                    # Freeze wheel during ghost: spring ON, inputs locked, CLAMP ON, low force to reduce buzz
                    _lock_wheel(ctrl_cached, spring=True)
                    if hasattr(ctrl_cached, "set_clamp"):
                        try: ctrl_cached.set_clamp(True)
                        except Exception: pass
                    try:
                        if hasattr(ctrl_cached, "autocenter_force"):
                            if force_orig is None:
                                force_orig = int(ctrl_cached.autocenter_force)
                            ctrl_cached.autocenter_force = int(ghost_force)
                            print(f"[PHASE] GHOST force -> {ctrl_cached.autocenter_force} (orig={force_orig})")
                    except Exception: pass
                    get_global_config().update({"autocenter_locked": True})

                    # If controller supports it, allow center updates in ghost
                    try:
                        if hasattr(ctrl_cached, "set_center_freeze"):
                            ctrl_cached.set_center_freeze(False)
                    except Exception:
                        pass

                elif phase == "prep":
                    # Pre-position window: inputs free, clamp OFF, spring ON but force=0 (free wheel)
                    _unlock_inputs_only(ctrl_cached)
                    if hasattr(ctrl_cached, "set_clamp"):
                        try: ctrl_cached.set_clamp(False)
                        except Exception: pass
                    try:
                        if hasattr(ctrl_cached, "autocenter_force"):
                            if force_orig is None:
                                force_orig = int(ctrl_cached.autocenter_force)
                            ctrl_cached.autocenter_force = 0
                            print(f"[PHASE] PREP force -> 0 (orig={force_orig})")
                    except Exception: pass
                    get_global_config().update({"autocenter_locked": False})
                    manual_force_curr = 0.0  # reset EMA before manual

                    # In prep we allow the center to float to true middle
                    try:
                        if hasattr(ctrl_cached, "set_center_freeze"):
                            ctrl_cached.set_center_freeze(False)
                    except Exception:
                        pass

                elif phase == "manual":
                    # Manual: keep spring ON, clamp OFF; dynamic force (speed & hysteresis & damping)
                    # Unlock EVERYTHING for free driving
                    if hasattr(ctrl_cached, "lock_input"):
                        try: 
                            ctrl_cached.lock_input(False)
                            print("[PHASE] MANUAL inputs unlocked")
                        except Exception: pass
                    
                    if hasattr(ctrl_cached, "set_clamp"):
                        try: 
                            ctrl_cached.set_clamp(False)
                            print("[PHASE] MANUAL clamp disabled")
                        except Exception: pass
                    
                    # Spring ON for autocenter feel (will have dynamic force based on speed/position)
                    if hasattr(ctrl_cached, "set_spring"):
                        try: 
                            ctrl_cached.set_spring(True)
                            print("[PHASE] MANUAL spring enabled (dynamic force)")
                        except Exception: pass
                    
                    # Policy-level: no steering override
                    get_global_config().update({"autocenter_locked": False})
                    print("[PHASE] MANUAL autocenter_locked=False (policy will not zero steering)")

                    # Freeze center (if controller supports) to avoid drifting baseline while user holds straight
                    try:
                        if hasattr(ctrl_cached, "set_center_freeze"):
                            ctrl_cached.set_center_freeze(True)
                    except Exception:
                        pass

                last_phase = phase

            # ---- Dynamic MANUAL force (speed-based + center hysteresis + damping) ----
            if phase == "manual" and ctrl_cached is not None and hasattr(ctrl_cached, "autocenter_force"):
                # vehicle speed
                try:
                    speed_kmh = float(getattr(env.agent, "speed_km_h", 0.0) or 0.0)
                except Exception:
                    speed_kmh = 0.0
                s = max(0.0, min(1.0, (speed_kmh / max(1e-6, speed_max)) ** shape_p))
                base_speed_force = manual_min + (manual_max - manual_min) * s

                # low-speed grace
                if speed_kmh <= low_spd_kmh:
                    base_speed_force *= low_spd_gain

                # read center error
                try:
                    err_deg_signed = float(ctrl_cached.get_center_error())
                except Exception:
                    err_deg_signed = 0.0
                err_deg = abs(err_deg_signed)

                # hysteresis state update
                if getattr(_manual_center_state, "inside", True):
                    if err_deg >= dead_out_deg:
                        _manual_center_state.inside = False
                else:
                    if err_deg <= dead_in_deg:
                        _manual_center_state.inside = True

                # center factor with hysteresis & progressive exponential ramp
                if _manual_center_state.inside:
                    center_factor = 0.0
                else:
                    if err_deg >= soft_out_deg:
                        center_factor = 1.0
                    else:
                        # Normalized position in ramp range [0, 1]
                        t = (err_deg - dead_out_deg) / max(1e-6, (soft_out_deg - dead_out_deg))
                        t = max(0.0, min(1.0, t))
                        # Apply exponential curve: stronger pull when far, gentler near center
                        center_factor = t ** center_exponent

                # Offset-dependent spring gain (matches enhanced_wheel offset_gain logic)
                # This makes force stronger when far from center (1.0 at center → 1.5 at full deflection)
                offset_norm = min(1.0, err_deg / max(1e-6, soft_out_deg))
                offset_gain = 1.0 + (1.5 - 1.0) * (offset_norm ** 1.0)  # offset_gain_max=1.5, linear ramp

                # damping term (deg/s)
                now_t = time.monotonic()
                dt = max(1e-3, now_t - getattr(_manual_center_state, "prev_t", now_t))
                derr_dt = (err_deg - abs(getattr(_manual_center_state, "prev_err", 0.0))) / dt
                _manual_center_state.prev_err = err_deg_signed
                _manual_center_state.prev_t   = now_t
                damp_term = kd_damp * min(abs(derr_dt), damp_cap)

                # Offset-dependent damping reduction (makes return FASTER when far from center)
                # At center (offset_norm=0): damp_reduction=1.0 (full damping, smooth & controlled)
                # At full lock (offset_norm=1): damp_reduction=0.25 (75% less damping, fast snap-back)
                damp_reduction = 1.0 - (0.75 * (offset_norm ** 0.9))  # Progressive damping reduction
                damp_term = damp_term * damp_reduction

                # Speed-sensitive steering resistance (makes steering heavier at high speeds)
                # This is ADDITIVE base resistance, not affected by center position
                speed_resistance = 0.0
                if speed_kmh > resist_thresh:
                    # Normalize speed above threshold
                    speed_factor = min(1.0, ((speed_kmh - resist_thresh) / max(1e-6, (speed_max - resist_thresh))) ** resist_shape)
                    speed_resistance = resist_min + (resist_max - resist_min) * speed_factor
                
                # Combine: (base_force × center_factor × offset_gain) + speed_resistance - damping
                # This matches enhanced_wheel: force_ = (offset^exp) × off_gain × speed_gain × base_force
                target_force = max(0.0, base_speed_force * center_factor * offset_gain + speed_resistance - damp_term)

                # EMA smooth
                _manual_center_state.ema = (1.0 - alpha) * getattr(_manual_center_state, "ema", target_force) + alpha * target_force
                try:
                    ctrl_cached.autocenter_force = int(max(0.0, min(32767.0, _manual_center_state.ema)))
                except Exception:
                    pass

            # HUD in replay window
            if not takeover_flag and frames_to_onset > 0:
                if frames_to_onset <= show_window_frames:
                    secs_left = max(0, int(round(frames_to_onset / float(fps))))
                    try:
                        current_speed = float(getattr(env.agent, "speed_km_h", 0.0) or 0.0)
                    except Exception:
                        current_speed = 0.0
                    screen.render(
                        f"📋 Scenario {current_scenario_num}/{total_configs} | ID: {sid}\n"
                        f"🚗 Speed: {current_speed:.1f} km/h\n"
                        "Currently playing loaded trajectory.\n"
                        f"Instruction:\n {instr}\n"
                        f"Manual takeover in: {secs_left}s\n"
                    )

            # MANUAL rising edge bookkeeping
            if takeover_flag and not prev_takeover_flag:
                takeover_start_index = i
            prev_takeover_flag = takeover_flag

            # sample screenshot a few frames after takeover
            if takeover_flag and not sampled_png and i > onset + 5:
                sampled_png = True
                try:
                    frame = engine._get_window_image()
                    sample_dir = os.path.join(samples_dir, f"samples_{seed}")
                    os.makedirs(sample_dir, exist_ok=True)
                    Image.fromarray(frame).save(os.path.join(sample_dir, f"sample_index_{i}.png"))
                except Exception:
                    pass

            # ----- record ego state -----
            try: positions.append(env.agent.position.tolist())
            except Exception: positions.append([0.0, 0.0])

            try: velocities.append(env.agent.velocity.tolist())
            except Exception: velocities.append([0.0, 0.0])

            # heading vector + theta
            theta_val = None
            head_vec = None
            try:
                if hasattr(env.agent, "heading_theta"):
                    theta_val = float(env.agent.heading_theta)
                elif hasattr(env.agent, "heading") and isinstance(env.agent, (int, float)):
                    theta_val = float(env.agent.heading)
            except Exception:
                theta_val = None

            try:
                if hasattr(env.agent, "heading") and hasattr(env.agent.heading, "tolist"):
                    head_vec = env.agent.heading.tolist()
                elif isinstance(env.agent.heading, (list, tuple)):
                    head_vec = list(env.agent.heading)
            except Exception:
                head_vec = None

            if head_vec is None and theta_val is not None:
                head_vec = [math.cos(theta_val), math.sin(theta_val)]

            headings.append(head_vec)
            heading_theta.append(theta_val)

            # raw controller action
            if info.get("raw_action"):
                steering, accel = info["raw_action"][0], info["raw_action"][1]
            else:
                steering, accel = None, None
            steering_signals.append(steering)
            accelerations.append(accel)

            # timeline
            is_manual_flags.append(takeover_flag)
            frames.append(i)
            try:
                t_now = engine.get_global_time()
                timestamps.append(float(t_now - t0))
            except Exception:
                timestamps.append(float(i))

            # generic HUD if not in countdown window
            if not (not takeover_flag and frames_to_onset > 0 and frames_to_onset <= show_window_frames):
                # Get current speed for display
                try:
                    current_speed = float(getattr(env.agent, "speed_km_h", 0.0) or 0.0)
                except Exception:
                    current_speed = 0.0
                
                if takeover_flag:
                    bad_marker = " [MARKED BAD]" if marked_bad else ""
                    msg = (
                        f"📋 Scenario {current_scenario_num}/{total_configs} ({scenarios_remaining} left) | ID: {sid}\n"
                        f"🚗 Speed: {current_speed:.1f} km/h\n"
                        f"{instr}{bad_marker}\n"
                        " Use W/S for accel/brake; A/D to steer.\n"
                        " Press D to mark bad config; X if impossible; Z if finished early."
                    )
                else:
                    msg = (
                        f"📋 Scenario {current_scenario_num}/{total_configs} ({scenarios_remaining} left) | ID: {sid}\n"
                        f"🚗 Speed: {current_speed:.1f} km/h\n"
                        "Currently playing loaded trajectory.\n"
                        " Prepare to take over shortly.\n"
                        f" Instructions will be:\n {instr}"
                    )
                screen.render(msg)

            # termination
            if tm or tc:
                term_reason = "time_or_terminate"
                break

        # cleanup HUD + wheel close-only
        try: screen.clear()
        except Exception: pass

        # Close wheel and restore original force
        _close_wheel_after_scenario(env, restore_force=force_orig)

        # SAVE - Enhanced validation and backup logic
        should_save = True
        save_reason = ""
        
        # Validation checks
        if bad_scenario:
            should_save = False
            save_reason = f"bad_scenario (term_reason={term_reason})"
        elif len(frames) == 0:
            should_save = False
            save_reason = "no frames recorded"
        elif takeover_start_index is None:
            should_save = False
            save_reason = "manual takeover never started"
        else:
            # Count how many manual frames we have
            manual_frame_count = sum(1 for flag in is_manual_flags if flag)
            if manual_frame_count == 0:
                should_save = False
                save_reason = "no manual control frames recorded"
            elif manual_frame_count < 5:
                print(f"⚠️ [Collect] Warning: Only {manual_frame_count} manual frames recorded (very short session)")
        
        if should_save:
            # Check if file exists and back it up
            if os.path.exists(output_path):
                backup_path = output_path.replace(".json", "_backup.json")
                try:
                    import shutil
                    shutil.copy2(output_path, backup_path)
                    print(f"[Collect] Existing file backed up to {backup_path}")
                except Exception as e:
                    print(f"⚠️ [Collect] Could not backup existing file: {e}")
            
            traj = {
                "scenario_id": sid,
                "seed": int(seed),
                "controller": controller,
                "selected_instruction": instr,
                "selected_onset_manual": int(onset_manual),
                "selected_instruction_index": int(instr_idx),

                "collect_start_index": int(collect_start_index),
                "takeover_start_index": int(takeover_start_index),

                "frames": frames,
                "timestamps": timestamps,
                "is_manual": is_manual_flags,

                "positions": positions,
                "velocities": velocities,
                "headings": headings,
                "heading_theta": heading_theta,

                "steering_signals": steering_signals,
                "accelerations": accelerations,

                "termination": term_reason,
                "final_index": frames[-1],
            }
            
            # Validate data integrity before saving
            try:
                # Check all arrays have same length
                expected_len = len(frames)
                arrays_to_check = {
                    "timestamps": timestamps,
                    "is_manual": is_manual_flags,
                    "positions": positions,
                    "velocities": velocities,
                    "headings": headings,
                    "heading_theta": heading_theta,
                    "steering_signals": steering_signals,
                    "accelerations": accelerations,
                }
                
                for name, arr in arrays_to_check.items():
                    if len(arr) != expected_len:
                        print(f"⚠️ [Collect] Data integrity warning: {name} has {len(arr)} elements, expected {expected_len}")
                
                # Save the file
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(traj, f, indent=1)
                
                # Verify file was written
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    manual_count = sum(1 for flag in is_manual_flags if flag)
                    file_size = os.path.getsize(output_path) / 1024  # KB
                    
                    print("\n" + "="*80)
                    print("💾 FILE SAVED SUCCESSFULLY!")
                    print("="*80)
                    print(f"✅ Scenario ID: {sid}")
                    print(f"✅ Controller: {controller}")
                    print(f"✅ File path: {output_path}")
                    print(f"✅ File size: {file_size:.1f} KB")
                    print(f"✅ Total frames: {len(frames)}")
                    print(f"✅ Manual frames: {manual_count}")
                    print(f"✅ Termination: {term_reason}")
                    print("="*80 + "\n")
                else:
                    print("\n" + "="*80)
                    print("❌ FILE SAVE FAILED!")
                    print("="*80)
                    print(f"❌ File path: {output_path}")
                    print(f"❌ File exists: {os.path.exists(output_path)}")
                    if os.path.exists(output_path):
                        print(f"❌ File size: {os.path.getsize(output_path)} bytes")
                    print("="*80 + "\n")
                    
            except Exception as e:
                print(f"❌ [Collect] Error saving trajectory: {e}")
                print(traceback.format_exc())
        else:
            print("\n" + "="*80)
            print("⚠️ SCENARIO NOT SAVED")
            print("="*80)
            print(f"⚠️ Scenario ID: {sid}")
            print(f"⚠️ Reason: {save_reason}")
            print(f"⚠️ File would be: {output_path}")
            print("="*80 + "\n")
        
        # Increment counter for next scenario
        scenarios_completed += 1

    env.close()
    return non_executables

# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metadrive Hybrid Data Collection (Simple Pipeline)")
    parser.add_argument("-c", "--config_path", default="configs", type=str, help="Path to the config files.")
    parser.add_argument("-d", "--data_path", default=waymo_data, type=str, help="Path to the dataset.")
    parser.add_argument("--replay", action="store_true", help="Replay previously collected inputs.")
    parser.add_argument("--controller", type=str, default="keyboard",
                        help="Input device: keyboard | xbox | wheel | enhanced_wheel")
    parser.add_argument("--out_dir", type=str, default="manual_inputs", help="Where to write collected JSON.")
    parser.add_argument("--inputs_dir", type=str, default=None,
                        help="Where replay reads JSON (default = --out_dir).")
    parser.add_argument("--samples_dir", type=str, default="samples", help="Where to write sample PNGs.")
    parser.add_argument("--only_seed", type=int, default=None, help="Run only this dataset seed (0-based).")
    parser.add_argument("--only_sid", type=str, default=None, help="Run only this scenario ID (e.g. '6e65895fb83fc9d5').")
    parser.add_argument("--onset_lead", type=int, default=7, help="Frames to lead before onset (main-branch behavior).")
    parser.add_argument("--topdown", action="store_true", help="Use topdown renderer during replay (interactive viewer).")
    parser.add_argument("--skip_ghost", action="store_true", help="Skip original trajectory, replay only human inputs (start from takeover).")

    args = parser.parse_args()

    if args.inputs_dir is None:
        args.inputs_dir = args.out_dir or "manual_inputs"

    try:
        config_files = [
            os.path.join(args.config_path, f)
            for f in os.listdir(args.config_path)
            if f.endswith(".json")
        ]
    except Exception:
        config_files = None
        print("Config folder does not exist")

    # REQUIRE configs to run (unchanged behavior)
    if config_files and not args.replay:
        collect(
            configs=config_files,
            data=args.data_path,
            controller=args.controller,
            out_dir=args.out_dir,
            samples_dir=args.samples_dir,
            only_seed=args.only_seed,
            only_sid=args.only_sid,
            onset_lead=args.onset_lead,
        )
    elif config_files and args.replay:
        replay(
            configs=config_files,
            data=args.data_path,
            controller=args.controller,
            inputs_dir=args.inputs_dir,
            only_seed=args.only_seed,
            only_sid=args.only_sid,
            onset_lead=args.onset_lead,
            use_topdown=args.topdown,
            skip_ghost=args.skip_ghost,
        )
    else:
        print("❌ No config JSONs found. Place .json files in --config_path and re-run.")


    
