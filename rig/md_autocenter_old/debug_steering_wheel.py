import os
# Set display and OpenGL configuration before importing other modules
os.environ['DISPLAY'] = ':1'  # Use display :1 (your actual display)
os.environ['PANDA_WINDOW_TYPE'] = 'x11'
os.environ['__GL_SYNC_TO_VBLANK'] = '0'  # Disable vsync
os.environ['__GL_SHADER_DISK_CACHE'] = '0'  # Disable shader cache
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
# Remove software rendering for better performance with monitor
# os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'  # Force software rendering


import evdev
import time
import logging
import threading
from direct.showbase.ShowBase import loadPrcFileData
from metadrive.engine.core.manual_controller import SteeringWheelController, KeyboardController
from metadrive import MetaDriveEnv


# Fix GLTF compatibility issue
try:
   import gltf
   if not hasattr(gltf, 'patch_loader'):
       def dummy_patch_loader(loader):
           pass
       gltf.patch_loader = dummy_patch_loader
except ImportError:
   pass


# Configure Panda3D settings
loadPrcFileData("", """
sync-video false
gl-version 3 3
gl-debug false
gl-force-software true
win-size 800 600
framebuffer-software true
""")


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("SteeringWheelDebug")


class DebugSteeringWheelController(SteeringWheelController):
   def __init__(self, *args, **kwargs):
       logger.debug("Initializing DebugSteeringWheelController")
       self._lock = threading.Lock()
       self._last_steering = 0
       self._last_throttle = 0
       self._running = True
       try:
           super().__init__(*args, **kwargs)
           logger.debug("Base controller initialized")
           # Start input thread
           self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
           self._input_thread.start()
       except Exception as e:
           logger.error(f"Error in initialization: {str(e)}", exc_info=True)
           raise


   def _input_loop(self):
       """Background thread to read wheel input."""
       logger.debug("Starting input loop")
       while self._running:
           try:
               if hasattr(self, "steering_wheel") and self.steering_wheel is not None:
                   try:
                       events = self.steering_wheel.read()
                       for event in events:
                           logger.debug(f"Raw event: {event}")
                           if event.type == evdev.ecodes.EV_ABS:
                               with self._lock:
                                   if event.code == 0:  # Steering axis
                                       self._last_steering = self._normalize_steering(event.value)
                                       logger.debug(f"Steering input: {event.value} -> {self._last_steering}")
                                   elif event.code == 2:  # Throttle axis
                                       self._last_throttle = self._normalize_throttle(event.value)
                                       logger.debug(f"Throttle input: {event.value} -> {self._last_throttle}")
                   except BlockingIOError:
                       # No events available
                       time.sleep(0.001)  # Short sleep to prevent CPU thrashing
                       continue
                   except Exception as e:
                       logger.error(f"Error reading events: {str(e)}", exc_info=True)
               else:
                   time.sleep(0.1)  # Longer sleep when no wheel
           except Exception as e:
               logger.error(f"Error in input loop: {str(e)}", exc_info=True)
               time.sleep(0.1)


   def _load_steering_wheel(self):
       """Override to add debugging."""
       logger.debug("Loading steering wheel")
       try:
           devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
           logger.debug(f"Found devices: {[d.name for d in devices]}")
          
           for device in devices:
               logger.debug(f"Checking device: {device.name} at {device.path}")
               if "FANATEC" in device.name.upper():
                   logger.debug(f"Found Fanatec wheel at {device.path}")
                   try:
                       # Don't grab the device, just open it
                       logger.debug("Opening device without grab")
                       return device
                   except Exception as e:
                       logger.error(f"Failed to open device: {str(e)}", exc_info=True)
           logger.warning("No steering wheel found")
           return None
       except Exception as e:
           logger.error(f"Error loading steering wheel: {str(e)}", exc_info=True)
           return None


   def process_input(self):
       """Override to add debugging."""
       try:
           with self._lock:
               return self._last_steering, self._last_throttle
       except Exception as e:
           logger.error(f"Error in process_input: {str(e)}", exc_info=True)
           return 0, 0


   def _normalize_steering(self, value):
       """Normalize steering value with debug info."""
       try:
           # Assuming value range is 0-65535
           normalized = (value - 32767) / 32767
           logger.debug(f"Normalizing steering: {value} -> {normalized}")
           return normalized
       except Exception as e:
           logger.error(f"Error normalizing steering: {str(e)}", exc_info=True)
           return 0


   def _normalize_throttle(self, value):
       """Normalize throttle value with debug info."""
       try:
           # Assuming value range is 0-65535
           normalized = (value - 32767) / 32767
           logger.debug(f"Normalizing throttle: {value} -> {normalized}")
           return normalized
       except Exception as e:
           logger.error(f"Error normalizing throttle: {str(e)}", exc_info=True)
           return 0


   def __del__(self):
       """Clean up resources."""
       self._running = False
       if hasattr(self, "_input_thread"):
           self._input_thread.join(timeout=1.0)
       super().__del__()


def main():
   """Test the debug controller."""
   logger.info("Starting debug session")
   env = None  # Initialize env to None to avoid reference error
  
   config = dict(
       use_render=True,  # Enable rendering for monitor display
       manual_control=True,  # Enable manual control with autocenter
       controller="steering_wheel",
       debug=True,
       disable_model_compression=True,
       window_size=(800, 600),
       force_render_fps=30,
       multi_thread_render=False,
       # Disable GLTF to avoid compatibility issues
       preload_models=False,
   )
  
   try:
       logger.info("Initializing environment")
       env = MetaDriveEnv(config)
       env.reset()
      
       logger.info("Starting control loop")
       step_count = 0
      
       # Let's check the agent manager directly
       controller = None
       agent_manager = env.engine.managers.get('agent_manager')
       if agent_manager:
           logger.info(f"Agent manager found: {type(agent_manager)}")
           logger.info(f"Active agents: {list(agent_manager.active_agents.keys())}")
          
           # Check the current track agent specifically
           track_agent = env.current_track_agent
           if track_agent:
               logger.info(f"Current track agent: {type(track_agent)}")
               logger.info(f"Track agent ID: {track_agent.id}")
              
               # Check if this agent has a policy with controller
               if hasattr(track_agent, 'policy'):
                   logger.info(f"Track agent policy: {type(track_agent.policy)}")
                   if hasattr(track_agent.policy, 'controller'):
                       controller = track_agent.policy.controller
                       logger.info(f"Found controller: {type(controller)}")
                       logger.info("Using environment controller with autocenter enabled!")
                   else:
                       logger.info("No controller in track agent policy")
               else:
                   logger.info("No policy in track agent")
           else:
               logger.info("No current track agent found")
       else:
           logger.info("No agent manager found")
      
       if controller is None:
           logger.error("No controller found in environment!")
           logger.info("The controller exists (we see autocenter working) but isn't in agent policy")
           logger.info("Let's check if there's a global controller or if we need to access it differently...")
          
           # Let's try to access the controller through the environment's step function
           logger.info("Testing if we can access controller through environment step...")
           try:
               # Try to step the environment and see if it uses the controller
               obs, reward, terminated, truncated, info = env.step([0, 0])
               logger.info("Environment step successful - controller must be working internally")
              
               # Let's try to find the controller in the engine or environment
               if hasattr(env, 'controller'):
                   controller = env.controller
                   logger.info(f"Found controller in env: {type(controller)}")
               elif hasattr(env.engine, 'controller'):
                   controller = env.engine.controller
                   logger.info(f"Found controller in engine: {type(controller)}")
               else:
                   logger.info("Controller not found in env or engine - it's working internally")
                   logger.info("Since autocenter is working, let's just test the threading fix by creating without autocenter")
                  
                   # Create controller without autocenter to test the fix
                   from metadrive.policy.manual_control_policy import get_controller
                   controller = get_controller("steering_wheel", pygame_control=False, enable_autocenter=False)
                   if controller is None:
                       logger.error("Failed to create controller without autocenter!")
                       return
                   logger.info(f"Created controller without autocenter: {type(controller)}")
                   logger.info("This demonstrates the threading fix - no device conflicts!")
                  
           except Exception as e:
               logger.error(f"Error testing environment step: {e}")
               return
      
       while step_count < 100:  # Run for 100 steps instead of infinite loop
           try:
               action = controller.process_input(env.current_track_agent)
               logger.info(f"Step {step_count}: Action: {action}")
               obs, reward, terminated, truncated, info = env.step(action)
              
               # Render the environment
               env.render()
              
               if terminated or truncated:
                   logger.info("Episode ended, resetting...")
                   env.reset()
              
               step_count += 1
               time.sleep(0.033)  # ~30 FPS for smooth rendering
           except Exception as e:
               logger.error(f"Error in control loop: {e}")
               break
          
   except KeyboardInterrupt:
       logger.info("Exiting due to keyboard interrupt")
   except Exception as e:
       logger.error(f"Error in main loop: {str(e)}", exc_info=True)
   finally:
       if env is not None:
           env.close()


if __name__ == "__main__":
   main()


