import json
import math
import logging

from metadrive.policy.env_input_policy import ExtraEnvInputPolicy
from metadrive.engine.engine_utils import get_global_config
from metadrive.policy.manual_control_policy import get_controller
from metadrive.scenario.parse_object_state import parse_object_state

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ====================== Base (NO controller here) ======================
class _HybridBase(ExtraEnvInputPolicy):
    """Shared helpers for both live-collect and replay variants (no controller here)."""

    def __init__(self, obj, seed):
        super().__init__(obj, seed)
        self._velocity_local_frame = False
        self.config_manual = None
        self.takeover = False
        self.action_info = {}
        self._takeover_armed = False
        self.traj_info = self._build_traj_cache()

    def _build_traj_cache(self):
        scenario = self.engine.data_manager.current_scenario
        trajectory_data = scenario["tracks"]
        sdc_track_index = str(scenario["metadata"]["sdc_id"])
        ret = []
        for i in range(len(trajectory_data[sdc_track_index]["state"]["position"])):
            ret.append(parse_object_state(trajectory_data[sdc_track_index], i))
        return ret

    def _apply_replay_pose(self, agent_id, index):
        """Follow the dataset pose up to takeover (neutral action returned)."""
        obj = self.engine.agent_manager.get_agent(agent_id)
        index = max(0, min(index, len(self.traj_info) - 1))
        info = self.traj_info[index]
        if not bool(info["valid"]):
            return [0.0, 0.0]

        # If dataset provides controls, set them (harmless); otherwise just pose.
        if "throttle_brake" in info and hasattr(obj, "set_throttle_brake"):
            try:
                obj.set_throttle_brake(float(info["throttle_brake"].item()))
            except Exception:
                pass
        if "steering" in info and hasattr(obj, "set_steering"):
            try:
                obj.set_steering(float(info["steering"].item()))
            except Exception:
                pass

        try:
            obj.set_position(info["position"])
        except Exception:
            pass
        try:
            obj.set_velocity(info["velocity"], in_local_frame=self._velocity_local_frame)
        except Exception:
            pass
        try:
            obj.set_heading_theta(info["heading"])
        except Exception:
            pass
        try:
            obj.set_angular_velocity(info["angular_velocity"])
        except Exception:
            pass

        # Always return neutral action while we’re “ghosting”.
        return [0.0, 0.0]


# ====================== COLLECTION ======================
class HybridEgoCarPolicy(_HybridBase):
    """
    Live data collection:
      - Dataset replay (pose) until onset_manual.
      - From onset onward, read controller every frame.
      - Steering is forcibly zeroed whenever `autocenter_locked=True`.
      - Controller can be forced to (re)initialize early via gc["force_controller_init"]=True.
    """

    def __init__(self, obj, seed):
        super().__init__(obj, seed)
        self.controller = None

        cfg = get_global_config()
        # Create controller early when rendering (wheel / xbox / keyboard).
        try:
            if cfg.get("use_render", False):
                self.controller = get_controller(cfg.get("controller", "keyboard"), pygame_control=False)
        except Exception as e:
            logger.warning(f"[Collect] Controller init failed; using None: {e}")
            self.controller = None

        # Allow data_collection.py to force-create after reset.
        try:
            self.ensure_controller_loaded()
        except Exception:
            pass

    # ---- early/forced controller creation hook ----
    def ensure_controller_loaded(self):
        """
        Create (or re-create) the controller if missing, or if a force flag is set.
        Lets data_collection.py bring the wheel up immediately after env.reset().
        """
        cfg = get_global_config()
        force = bool(cfg.get("force_controller_init", False))
        if self.controller is not None and not force:
            return

        self.controller = None
        try:
            if cfg.get("use_render", False):
                self.controller = get_controller(cfg.get("controller", "keyboard"), pygame_control=False)
        except Exception as e:
            logger.warning(f"[Collect] ensure_controller_loaded() failed; using None: {e}")
            self.controller = None

        # clear one-shot flag
        try:
            cfg["force_controller_init"] = False
        except Exception:
            pass

    def act(self, agent_id):
        gc = get_global_config()
        if self.config_manual is None:
            # `onset_manual` is treated as a FRAME index here.
            self.config_manual = int(gc.get("onset_manual", 0))

        index = int(self.engine.external_actions[agent_id]["extra"])
        ctrl = self.controller

        # --- Stage 2 (GHOST) until onset: replay dataset pose, neutral action ---
        if index < self.config_manual:
            return self._apply_replay_pose(agent_id, index)

        # --- First frame at/after onset: raise takeover flag (for HUD/recorder) ---
        if not self._takeover_armed:
            self._takeover_armed = True
            self.takeover = True
            try:
                self.engine.global_config.update({"takeover": True})
            except Exception:
                pass

        # --- Manual control path (Stage 4) ---
        expert_action = [0.0, 0.0]
        if ctrl is not None and hasattr(ctrl, "process_input"):
            try:
                # Pass vehicle to keep haptics fed and allow speed-based effects.
                expert_action = ctrl.process_input(self.engine.current_track_agent)
            except Exception as e:
                logger.warning(f"[Collect] process_input failed; falling back to zeros: {e}")
                expert_action = [0.0, 0.0]

        # Belt & suspenders: while data_collection keeps the wheel locked,
        # force steering=0 so the sim can't twitch even if controller leaks.
        if bool(gc.get("autocenter_locked", False)):
            expert_action[0] = 0.0

        self.action_info["raw_action"] = expert_action  # consumed by the recorder
        return expert_action


# ====================== REPLAY (no controller; use saved data) ======================
class HybridEgoCarReplayPolicy(_HybridBase):
    """
    Replay:
      - Before onset → dataset pose replay (ghost).
      - After onset:
          * If gc["replay_use_states"]=True → set recorded ego state each frame (exact ghost).
          * Else → apply recorded [steer, throttle_brake] actions.
      - No controller is constructed/used here.
    """

    def __init__(self, obj, seed):
        super().__init__(obj, seed)
        self.manual_inputs = None
        self.loaded_inputs = None
        self._cached_action = [0.0, 0.0]
        self._use_states = False
        self._offset = 0  # alignment between onset and first saved frame

    def reset(self):
        cfg = get_global_config()
        self.manual_inputs = cfg.get("manual_inputs", None)
        self._use_states = bool(cfg.get("replay_use_states", False))
        onset = int(cfg.get("onset_manual", 0))
        self._offset = 0

        if self.manual_inputs:
            try:
                with open(self.manual_inputs, "r") as f:
                    self.loaded_inputs = json.load(f)

                # Align indices: prefer collect_start_index, else takeover_start_index.
                cs = self.loaded_inputs.get("collect_start_index")
                to = self.loaded_inputs.get("takeover_start_index")
                if isinstance(cs, int):
                    self._offset = max(0, int(cs) - onset)
                elif isinstance(to, int):
                    self._offset = max(0, int(to) - onset)

                logger.info(
                    f"[Replay] Loaded {self.manual_inputs}; onset={onset} "
                    f"offset={self._offset} use_states={self._use_states}"
                )
            except Exception as e:
                logger.warning(f"[Replay] Could not load manual inputs: {e}")
                self.loaded_inputs = None

        return super().reset()

    def _apply_saved_state(self, agent_id, vi):
        if not self.loaded_inputs:
            return [0.0, 0.0]

        positions = self.loaded_inputs.get("positions", [])
        headings = self.loaded_inputs.get("headings", [])
        heading_theta = self.loaded_inputs.get("heading_theta", [])
        velocities = self.loaded_inputs.get("velocities", [])

        if vi >= len(positions) or vi >= len(velocities) or (
            len(headings) == 0 and len(heading_theta) == 0
        ):
            return [0.0, 0.0]

        obj = self.engine.agent_manager.get_agent(agent_id)
        pos = positions[vi]
        vel = velocities[vi]

        # Pose set
        try:
            if hasattr(obj, "set_position"):
                obj.set_position(pos)
        except Exception:
            pass

        # Heading: prefer theta array; else derive from heading vector.
        theta = None
        if vi < len(heading_theta) and heading_theta[vi] is not None:
            try:
                theta = float(heading_theta[vi])
            except Exception:
                theta = None
        if theta is None and vi < len(headings):
            head = headings[vi]
            if isinstance(head, (int, float)):
                theta = float(head)
            elif isinstance(head, (list, tuple)) and len(head) == 2:
                try:
                    theta = math.atan2(float(head[1]), float(head[0]))
                except Exception:
                    theta = None
            elif isinstance(head, (list, tuple)) and head and isinstance(head[0], (int, float)):
                theta = float(head[0])

        try:
            if theta is not None and hasattr(obj, "set_heading_theta"):
                obj.set_heading_theta(theta)
        except Exception:
            pass

        try:
            if hasattr(obj, "set_velocity"):
                obj.set_velocity(vel, in_local_frame=False)
        except Exception:
            pass

        # Keep last action cached for info consumers.
        steer_list = self.loaded_inputs.get("steering_signals", [])
        acc_list = self.loaded_inputs.get("accelerations", [])
        if vi < len(steer_list) and vi < len(acc_list):
            self._cached_action = [steer_list[vi], acc_list[vi]]

        # Neutral action while we directly set states.
        return [0.0, 0.0]

    def act(self, agent_id):
        gc = get_global_config()
        if self.config_manual is None:
            self.config_manual = int(gc.get("onset_manual", 0))

        index = int(self.engine.external_actions[agent_id]["extra"])

        # Pre-onset → pure dataset ghost.
        if index < self.config_manual:
            return self._apply_replay_pose(agent_id, index)

        # After onset → replay from saved JSON.
        if not self.loaded_inputs:
            try:
                gc.update({"replay_control": False})
            except Exception:
                pass
            return [0.0, 0.0]

        try:
            gc.update({"replay_control": True})
        except Exception:
            pass

        # Align into saved arrays
        vi = max(0, index - self.config_manual - int(getattr(self, "_offset", 0)))

        if self._use_states:
            # Pin physics while setting exact states (if supported)
            obj = self.engine.agent_manager.get_agent(agent_id)
            if hasattr(obj, "set_static"):
                try:
                    obj.set_static(True)
                except Exception:
                    pass
            return self._apply_saved_state(agent_id, vi)

        # Action replay path
        steer_list = self.loaded_inputs.get("steering_signals", [])
        acc_list = self.loaded_inputs.get("accelerations", [])
        if vi < len(steer_list) and vi < len(acc_list):
            self._cached_action = [steer_list[vi], acc_list[vi]]

        # If someone left autocenter lock on during replay, keep steering neutral.
        if bool(gc.get("autocenter_locked", False)):
            self._cached_action[0] = 0.0

        return self._cached_action



