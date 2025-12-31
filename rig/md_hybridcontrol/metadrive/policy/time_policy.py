import logging
from metadrive.scenario.parse_object_state import parse_object_state
from metadrive.engine.engine_utils import get_global_config
from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController, XboxController
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy.manual_control_policy import get_controller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
JOYSTICK_DEADZONE = 0.025
class HybridEgoCarPolicy(EnvInputPolicy):
    def __init__(self, obj,seed):
        track = self.engine.data_manager.current_scenario["tracks"]
        super(HybridEgoCarPolicy, self).__init__(obj,seed)
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            self.controller = get_controller(config["controller"], pygame_control=False)
        self.takeover = False
        self.start_index = 0
        self._velocity_local_frame = False
        self.traj_info = self.get_trajectory_info(track)
        config = get_global_config()
        self.controller = get_controller(config["controller"], pygame_control=False)
        self.takeover = False
        self.config_manual = config["onset_manual"]
    @property
    def is_current_step_valid(self):
        return self.traj_info[self.episode_step] is not None
    def get_trajectory_info(self, trajectory):
        trajectory_data = self.engine.data_manager.current_scenario["tracks"]
        sdc_track_index = str(self.engine.data_manager.current_scenario["metadata"]["sdc_id"])
        ret = []
        for i in range(len(trajectory_data[sdc_track_index]["state"]["position"])):
            print(len(trajectory_data[sdc_track_index]["state"]["position"]))
            ret.append(parse_object_state(
                trajectory_data[sdc_track_index],
                i,
            ))
        return ret

    def act(self, agent_id):
      index = max(int(self.episode_step), 0)
    #  print("index from env",self.config_manual)
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
           if self.engine.global_config.get("set_static", False):
             self.obj.set_static(True)

           return None  
      else:
            self.engine.global_config["manual_control"] = True
            if self.engine.global_config["manual_control"]  :
              expert_action = self.controller.process_input(self.engine.current_track_agent)
            if isinstance(self.controller, SteeringWheelController) and (self.controller.left_shift_paddle
                                                                         or self.controller.right_shift_paddle):
                self.takeover = True
                self.engine.global_config["manual_control"] = False
                return expert_action
            elif isinstance(self.controller, KeyboardController) and (self.controller.takeover
                                                                      or abs(sum(expert_action)) > 0.01):
                self.takeover = True
                self.engine.global_config["manual_control"] = False
                return expert_action
            elif isinstance(self.controller, XboxController) and (self.controller.takeover or self.controller.button_a
                                                                     or self.controller.button_b or
                                                                  self.controller.button_x or self.controller.button_y
                                                                  or abs(sum(expert_action)) > JOYSTICK_DEADZONE):
                self.takeover = True
                self.engine.global_config["manual_control"] = False
                return expert_action
            
   
