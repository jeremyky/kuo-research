import logging
from metadrive.scenario.parse_object_state import parse_object_state
from metadrive.engine.engine_utils import get_global_config
from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController, XboxController
from metadrive.policy.env_input_policy import ExtraEnvInputPolicy
from metadrive.policy.manual_control_policy import get_controller
import gymnasium as gym
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
JOYSTICK_DEADZONE = 0.025
class HybridEgoCarPolicy(ExtraEnvInputPolicy):
    
    def __init__(self, obj,seed):
        self._haptics_active = False
        track = self.engine.data_manager.current_scenario["tracks"]
        super(HybridEgoCarPolicy, self).__init__(obj,seed)
        config = get_global_config()
        self.controller = get_controller(config["controller"], pygame_control=False)
        self.takeover = False
        self.start_index = 0
        self._velocity_local_frame = False
        self.traj_info = self.get_trajectory_info(track)
        config = get_global_config()
        self.takeover = False
        self.config_manual = None
        self._prep_freeze_frames = 200
        self._prep_until_index = None

        self._prep_freeze_frames = 600          # pre-freeze window before takeover (frames)
        self._prep_until_index = None
        self._center_eps = 0.02                # how close to 0 counts as centered ([-1..1] units)
        self._center_stable_frames = 10        # need this many consecutive frames inside eps
        self._center_max_wait_s = 5.0          # hard timeout so we don't hang forever
        self._center_stable_count = 0
        self._center_deadline_ts = None






    
    @property
    def is_current_step_valid(self):
        return self.traj_info[self.episode_step] is not None
    def get_trajectory_info(self, trajectory):
        trajectory_data = self.engine.data_manager.current_scenario["tracks"]
        sdc_track_index = str(self.engine.data_manager.current_scenario["metadata"]["sdc_id"])
        ret = []
        for i in range(len(trajectory_data[sdc_track_index]["state"]["position"])):
            ret.append(parse_object_state(
                trajectory_data[sdc_track_index],
                i,
            ))
        return ret
    
    

    def act(self, agent_id):
        if self.config_manual is None:
            self.config_manual = self.engine.global_config['onset_manual']
        index = int(self.engine.external_actions[agent_id]["extra"])
        ctrl = getattr(self, "controller", None)


        # ---------- arm pre-takeover freeze window ----------
        if (self._prep_until_index is None
            and index >= self.config_manual - self._prep_freeze_frames
            and index < self.config_manual):
            self._prep_until_index = self.config_manual
            self._center_stable_count = 0
            self._center_deadline_ts = time.time() + float(self._center_max_wait_s)


            # start haptics now and zero inputs so it centers while frozen
            try:
                if ctrl and hasattr(ctrl, "_lock") and hasattr(ctrl, "resume_autocenter"):
                    with ctrl._lock:
                        ctrl._shared["steering"] = 0.0
                        ctrl._shared["speed_kmh"] = 0.0
                    ctrl.resume_autocenter(delay_s=0.0)
                    self._haptics_active = True
            except Exception:
                pass


        # ---------- hold car and wait until wheel is centered (or timeout) ----------
        if self._prep_until_index is not None and index < self._prep_until_index:
            # freeze ego
            try:
                obj = self.engine.agent_manager.get_agent(agent_id)
                if hasattr(obj, "set_static"):
                    obj.set_static(True)
                info = self.traj_info[index]
                if bool(info["valid"]):
                    obj.set_position(info["position"])
                    obj.set_heading_theta(info["heading"])
                    obj.set_velocity([0.0, 0.0], in_local_frame=True)
                    obj.set_angular_velocity(0.0)
            except Exception:
                pass


            # check wheel center status
            centered = False
            try:
                if ctrl and hasattr(ctrl, "_lock"):
                    with ctrl._lock:
                        v = float(ctrl._shared.get("steering", 0.0))
                    centered = abs(v) <= float(self._center_eps)
            except Exception:
                centered = False


            if centered:
                self._center_stable_count += 1
            else:
                self._center_stable_count = 0


            # keep freezing until centered for N frames OR timeout reached
            if (self._center_stable_count < int(self._center_stable_frames)
                and (self._center_deadline_ts is None or time.time() < self._center_deadline_ts)):
                return None  # continue freezing
            # else fall through and let takeover start next block


        # ---------- end freeze exactly at takeover; unfreeze once ----------
        if self._prep_until_index is not None and index >= self._prep_until_index:
            try:
                obj = self.engine.agent_manager.get_agent(agent_id)
                if hasattr(obj, "set_static"):
                    obj.set_static(False)
            except Exception:
                pass
            self._prep_until_index = None
            self._center_deadline_ts = None
            self._center_stable_count = 0


        # ---------- haptics toggle: off during replay, on at/after takeover ----------
        try:
            if index < self.config_manual:
                if ctrl and hasattr(ctrl, "pause_autocenter") and self._haptics_active:
                    ctrl.pause_autocenter()
                    self._haptics_active = False
            else:
                if ctrl and hasattr(ctrl, "resume_autocenter") and not self._haptics_active:
                    if hasattr(ctrl, "_lock"):
                        with ctrl._lock:
                            ctrl._shared["steering"] = 0.0
                            ctrl._shared["speed_kmh"] = 0.0
                    ctrl.resume_autocenter(delay_s=0.0)
                    self._haptics_active = True
        except Exception:
            pass


        # ---------- normal replay / manual control ----------
        if index == self.config_manual:
            obj = self.engine.agent_manager.get_agent(agent_id)


        if index < self.config_manual:
            obj = self.engine.agent_manager.get_agent(agent_id)
            info = self.traj_info[index]
            if not bool(info["valid"]):
                return None
            if "throttle_brake" in info and hasattr(obj, "set_throttle_brake"):
                obj.set_throttle_brake(float(info["throttle_brake"].item()))
            if "steering" in info and hasattr(obj, "set_steering"):
                obj.set_steering(float(info["steering"].item()))
            obj.set_position(info["position"])
            obj.set_velocity(info["velocity"], in_local_frame=self._velocity_local_frame)
            obj.set_heading_theta(info["heading"])
            obj.set_angular_velocity(info["angular_velocity"])
            return None
        else:
            self.controller.takeover = True
            self.engine.global_config.update({'takeover': True})
            expert_action = self.controller.process_input(self.engine.current_track_agent)
            self.takeover = True
            self.action_info["raw_action"] = expert_action
            return expert_action






                
            
class HybridEgoCarReplayPolicy(ExtraEnvInputPolicy):
    
    def __init__(self, obj,seed):
        track = self.engine.data_manager.current_scenario["tracks"]
        super(HybridEgoCarReplayPolicy, self).__init__(obj,seed)
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            self.controller = get_controller(config["controller"], pygame_control=False)
        self.takeover = False
        self.start_index = 0
        self._velocity_local_frame = False
        self.traj_info = self.get_trajectory_info(track)
        config = get_global_config()
        #self.controller = get_controller(config["controller"], pygame_control=False)
        self.config_manual = None
        self.manual_inputs = None
        self._haptics_active = False  # track haptics state for clean toggling
        self._haptics_active = False
        self._prep_freeze_frames = 60
        self._prep_until_index = None






        
    @property
    def is_current_step_valid(self):
        return self.traj_info[self.episode_step] is not None
    def get_trajectory_info(self, trajectory):
        trajectory_data = self.engine.data_manager.current_scenario["tracks"]
        sdc_track_index = str(self.engine.data_manager.current_scenario["metadata"]["sdc_id"])
        ret = []
        for i in range(len(trajectory_data[sdc_track_index]["state"]["position"])):
            #print(len(trajectory_data[sdc_track_index]["state"]["position"]))
            ret.append(parse_object_state(
                trajectory_data[sdc_track_index],
                i,
            ))
        return ret
    
    
    def prepare_inputs(self):
        if 'manual_inputs' not in self.engine.global_config:
            raise ValueError('No Manual Input File Given')
        values = []
        with open(self.engine.global_config['manual_inputs'], 'r') as f:
            inputs = json.load(f)
        for idx, acc in enumerate(inputs['accelerations']):
            values.append([inputs['steering_signals'][idx], acc])
        return values


    def act(self, agent_id):
        if self.config_manual is None:
            self.config_manual = self.engine.global_config['onset_manual']
        index = int(self.engine.external_actions[agent_id]["extra"])


        ctrl = getattr(self, "controller", None)


        # ---------- arm a short pre-takeover freeze window ----------
        if (self._prep_until_index is None
            and index >= self.config_manual - self._prep_freeze_frames
            and index < self.config_manual):
            self._prep_until_index = self.config_manual
            # bring the spring up now and zero inputs so it centers while frozen
            try:
                if ctrl and hasattr(ctrl, "_lock") and hasattr(ctrl, "resume_autocenter"):
                    with ctrl._lock:
                        ctrl._shared["steering"] = 0.0
                        ctrl._shared["speed_kmh"] = 0.0
                    ctrl.resume_autocenter(delay_s=0.0)
                    self._haptics_active = True
            except Exception:
                pass


        # ---------- hold the car still during the freeze window ----------
        if self._prep_until_index is not None and index < self._prep_until_index:
            try:
                obj = self.engine.agent_manager.get_agent(agent_id)
                if hasattr(obj, "set_static"):
                    obj.set_static(True)
                info = self.traj_info[index]
                if bool(info["valid"]):
                    obj.set_position(info["position"])
                    obj.set_heading_theta(info["heading"])
                    obj.set_velocity([0.0, 0.0], in_local_frame=True)
                    obj.set_angular_velocity(0.0)
            except Exception:
                pass
            return None  # skip control while frozen


        # ---------- end freeze exactly at takeover ----------
        if self._prep_until_index is not None and index >= self._prep_until_index:
            try:
                obj = self.engine.agent_manager.get_agent(agent_id)
                if hasattr(obj, "set_static"):
                    obj.set_static(False)
            except Exception:
                pass
            self._prep_until_index = None


        # ---------- haptics toggle: off during replay, on at/after takeover ----------
        try:
            if index < self.config_manual:
                if ctrl and hasattr(ctrl, "pause_autocenter") and self._haptics_active:
                    ctrl.pause_autocenter()
                    self._haptics_active = False
            else:
                if ctrl and hasattr(ctrl, "resume_autocenter") and not self._haptics_active:
                    # ensure zero start on the exact takeover frame too
                    if hasattr(ctrl, "_lock"):
                        with ctrl._lock:
                            ctrl._shared["steering"] = 0.0
                            ctrl._shared["speed_kmh"] = 0.0
                    ctrl.resume_autocenter(delay_s=0.0)
                    self._haptics_active = True
        except Exception:
            pass


        # ---------- normal replay / manual control ----------
        if index == self.config_manual:
            obj = self.engine.agent_manager.get_agent(agent_id)


        if index < self.config_manual:
            obj = self.engine.agent_manager.get_agent(agent_id)
            info = self.traj_info[index]
            if not bool(info["valid"]):
                return None
            if "throttle_brake" in info and hasattr(obj, "set_throttle_brake"):
                obj.set_throttle_brake(float(info["throttle_brake"].item()))
            if "steering" in info and hasattr(obj, "set_steering"):
                obj.set_steering(float(info["steering"].item()))
            obj.set_position(info["position"])
            obj.set_velocity(info["velocity"], in_local_frame=self._velocity_local_frame)
            obj.set_heading_theta(info["heading"])
            obj.set_angular_velocity(info["angular_velocity"])
            return None
        else:
            self.controller.takeover = True
            self.engine.global_config.update({'takeover': True})
            expert_action = self.controller.process_input(self.engine.current_track_agent)
            self.takeover = True
            self.action_info["raw_action"] = expert_action
            return expert_action













        #print(index)
        if index == self.config_manual:
            obj=self.engine.agent_manager.get_agent(agent_id)
            #print("TAKEOVER STATE:", obj.position, obj.velocity, obj.heading)
        if index<self.config_manual:
           obj=self.engine.agent_manager.get_agent(agent_id)
           info = self.traj_info[index]
           if not bool(info["valid"]):
            return None  
           if "throttle_brake" in info:
            if hasattr(obj, "set_throttle_brake"):
                obj.set_throttle_brake(float(info["throttle_brake"].item()))
           if "steering" in info:
            if hasattr(obj, "set_steering"):
                obj.set_steering(float(info["steering"].item()))
           obj.set_position(info["position"])
           obj.set_velocity(info["velocity"], in_local_frame=self._velocity_local_frame)
           obj.set_heading_theta(info["heading"])
           obj.set_angular_velocity(info["angular_velocity"])
        #    if self.engine.global_config.get("set_static", False):
        #         self.obj.set_static(True)
           return None
        else:
            # if self.engine.global_config.get("set_static", True):
            #     self.obj.set_static(False)
            #self.engine.global_config['replay_control'] = True
            self.engine.global_config.update({'replay_control' : True})
            self.takeover = True
            if not self.manual_inputs:
                raise ValueError("No manual input file to read from.")
            value_index = index - self.config_manual
            return self.manual_inputs[value_index]

            
   