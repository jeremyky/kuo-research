
#!/usr/bin/env python3
"""
Simple test script for manual control with steering wheel and autocenter testing.
"""


import os
import sys
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.manual_control_policy import ManualControlPolicy


# Set display
os.environ['DISPLAY'] = ':1'


def test_manual_control():
   """Test manual control with steering wheel."""
  
   config = {
       "manual_control": True,
       "controller": "steering_wheel",
       "use_render": True,
       "window_size": (1200, 900),
       "preload_models": False,
       "disable_model_compression": True,
       "agent_policy": ManualControlPolicy,  # Use the manual control policy
   }
  
   print("Starting manual control test...")
   print(f"Display: {os.environ.get('DISPLAY', 'Not set')}")
  
   env = None
   try:
       env = MetaDriveEnv(config)
       print("Environment created successfully!")
      
       # Reset environment
       obs, info = env.reset()
       print("Environment reset successfully!")
       print("You should see a MetaDrive window now.")
       print("Use your steering wheel to control the car.")
       print("Press Ctrl+C to exit.")
      
       step_count = 0
       while step_count < 1000:  # Run for up to 1000 steps
           # For manual control, let the policy handle the input automatically
           # The ManualControlPolicy will process steering wheel input internally
           obs, reward, terminated, truncated, info = env.step(None)
          
           if terminated or truncated:
               print(f"Episode ended at step {step_count}")
               break
              
           step_count += 1
          
   except KeyboardInterrupt:
       print("\nManual control test interrupted by user.")
   except Exception as e:
       print(f"Error during manual control test: {e}")
       import traceback
       traceback.print_exc()
   finally:
       if env is not None:
           env.close()
           print("Environment closed.")


if __name__ == "__main__":
   test_manual_control()



# just testing if the steering wheel works, doesnt actually control ego car, need to test autocenter threading still