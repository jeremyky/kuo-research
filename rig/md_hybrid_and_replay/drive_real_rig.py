#!/usr/bin/env python
"""
Working version with correct axis mapping and autocreep logic from the old implementation
"""
import argparse
import os
import threading
import time


from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.engine.core.manual_controller import KeyboardController, XboxController
from enhanced_steering_wheel_controller_old import create_enhanced_steering_wheel_controller
from metadrive.policy.env_input_policy import EnvInputPolicy
from data_collection import clear_leaked_autocenter_config


# Enable rendering
os.environ["SDL_VIDEODRIVER"] = "x11"  # For Linux/Mac


RENDER_MESSAGE = {
   "1": "Switch to the main view",
   "2": "Switch to the top-down view",
   "3": "Switch to the third-person view",
   "4": "Switch to the first-person view",
   "5": "Switch to the free camera view",
   "6": "Switch to the back view",
   "7": "Switch to the left view",
   "8": "Switch to the right view",
   "9": "Switch to the front view",
   "0": "Switch to the back view",
   "q": "Switch to the top-down view",
   "b": "Switch to the top-down view",
   "c": "Switch to the third-person view",
   "r": "Reset the environment",
   "f": "Switch FPS to unlimited / realtime",
   "g": "Switch to the top-down view",
   "h": "Show help message",
   "i": "Switch to the top-down view",
   "j": "Switch to the top-down view",
   "k": "Switch to the top-down view",
   "l": "Switch to the top-down view",
   "m": "Switch to the top-down view",
   "n": "Switch to the top-down view",
   "o": "Switch to the top-down view",
   "p": "Switch to the top-down view",
   "s": "Switch to the top-down view",
   "t": "Switch to the top-down view",
   "u": "Switch to the top-down view",
   "v": "Switch to the top-down view",
   "w": "Switch to the top-down view",
   "x": "Switch to the top-down view",
   "y": "Switch to the top-down view",
   "z": "Switch to the top-down view",
   "esc": "Quit",
}




# Use the enhanced steering wheel controller from the separate module
# This provides modular autocenter functionality that can be reused in any script




def main():
   parser = argparse.ArgumentParser(description="MetaDrive Open World Driving with Steering Wheel")
   parser.add_argument("--controller", type=str, default="steering_wheel", choices=["steering_wheel", "xbox", "keyboard"], help="Controller type")
   parser.add_argument("--enable_autocenter", action="store_true", help="Enable force feedback autocenter")
   parser.add_argument("--autocenter_force", type=int, default=2000, help="Autocenter force strength (default: 2000)")
   parser.add_argument("--map", type=str, default="random", help="Map type (random, 7, 30, CCC, etc.)")
   parser.add_argument("--traffic_density", type=float, default=0.1, help="Traffic density (0.0 to 1.0)")
   parser.add_argument("--top_down", action="store_true", help="Use top-down view")
   parser.add_argument("--add_sensor", action="store_true", help="Add additional sensors")
   args = parser.parse_args()

   # Clear any leaked autocenter config from previous data_collection.py runs
   print("🧹 Clearing any leaked autocenter configuration...")
   clear_leaked_autocenter_config()

   print("🌍 Using MetaDrive Open World Environment")


   # Initialize controller using enhanced modular system
   controller = None
   if args.controller == "steering_wheel":
       try:
           print("🎮 Initializing enhanced steering wheel controller...")
           print(f"🎯 Autocenter force: {args.autocenter_force}")
           controller = create_enhanced_steering_wheel_controller(
               enable_autocenter=args.enable_autocenter,
               autocenter_force=args.autocenter_force
           )
           print("✅ Enhanced steering wheel controller initialized!")
       except Exception as e:
           print(f"❌ Failed to initialize steering wheel: {str(e)}")
           print("🔄 Falling back to keyboard controller...")
           controller = KeyboardController(pygame_control=True)
   elif args.controller == "xbox":
       controller = XboxController()
   else:
       controller = KeyboardController(pygame_control=True)


   if not controller:
       print("❌ Failed to initialize any controller. Exiting.")
       return


   # Environment configuration for open world - simplified
   cfg = {
       "manual_control": False,  # Disable manual control in environment
       "agent_policy": EnvInputPolicy,  # Use EnvInputPolicy to accept our actions
       "use_render": True if not args.top_down else False,
       "window_size": (1200, 800),
       "show_interface": True,
       "show_logo": True,
       "show_fps": True,
       # Basic open world settings
       "map": int(args.map) if args.map.isdigit() else 3,  # Map type (integer)
       "traffic_density": args.traffic_density,  # Traffic density
       "random_traffic": True,  # Enable random traffic
       # Make it run indefinitely
       "horizon": None,  # No step limit
       "out_of_road_penalty": 0.0,  # No penalty for going off-road
       "crash_vehicle_penalty": 0.0,  # No penalty for crashes
       "crash_object_penalty": 0.0,  # No penalty for object crashes
   }
  
   if args.add_sensor:
       additional_cfg = {
           "interface_panel": ["rgb_camera", "depth_camera", "semantic"],
           "sensors": {
               "rgb_camera": (RGBCamera, 1200, 800),
               "depth_camera": (DepthCamera, 1200, 800),
               "semantic_camera": (SemanticCamera, 1200, 800),
           },
       }
       cfg.update(additional_cfg)


   try:
       print("🚀 Initializing open world environment...")
       print(f"🔧 Configuration: {cfg}")
       env = MetaDriveEnv(cfg)
       print("✅ Open world environment initialized!")


       print("🎮 Starting interactive driving session...")
       print("Controls:")
       print("- Use your controller to drive")
       print("- ESC: Quit")
       print("- Q/B: Switch camera view")
       print("🔄 Environment will run FOREVER - no resets!")


       # Initialize the environment
       print("🔄 Initializing environment...")
       env.reset()
       print("✅ Environment loaded! Start driving!")


       step_count = 0


       # Main driving loop - runs FOREVER without resets
       while True:
           try:
               # Get action from our enhanced controller
               action = controller.process_input(env.agent)
              
               # Step the environment with our action
               o, r, tm, tc, info = env.step(action)
              
               # Render the scene
               env.render()


               step_count += 1


               # Debug output every 100 steps
               if step_count % 100 == 0:
                   print(f"🚗 Step {step_count}: Speed: {env.agent.speed_km_h:.1f} km/h, Position: {env.agent.position}")
                   print(f"🎯 Autocenter active: {hasattr(controller, 'shared_data') and controller.shared_data.get('running', False)}")


               # NEVER RESET - Keep driving indefinitely
               # The environment will continue running without any resets
              
           except Exception as step_error:
               print(f"⚠️ Step error (continuing): {step_error}")
               # Continue the loop even if there's an error in a single step
               continue


   except KeyboardInterrupt:
       print("\n🛑 Interrupted by user")
   except Exception as e:
       print(f"❌ Error during execution: {str(e)}")
       import traceback
       print(f"🔍 Full error traceback:")
       traceback.print_exc()
   finally:
       print("🧹 Cleanup completed.")
       if controller and hasattr(controller, 'cleanup'):
           controller.cleanup()
       if 'env' in locals():
           env.close()




if __name__ == "__main__":
   main()