##JSON Control Policy: By Nitya Mitnala
## Aim:
The goal is to add the capability to record a JSON file containing the scenario id, throttle information and steering angle information during Manual Keyboad Control. This JSON file is then used to recreate the same action using the recorded information. This is called JSON Control. The purpose is to combine this with the Hybrid Control Policy, so that the keyboard action during the user takeover can be recorded and used to recreate scenarios for training purposes.

## Changes made:
The following changes were made:
1. JSONController was added in manual_controller.py. This is a class that takes as input the path to the JSON file, and feeds the recorded values for throttle and steering to the action.
2. KeyboardController was modified with an added function that recodred the trajectory data into a list and returned the list.
3. ManualControlPolicy was changed to accommodate JSONController, if the controller was chosen to be "json", and KeyboardController otherwise. A function was added here also to return the recorded trajectory data, in case it was present, and if not, return NULL.
4. The main code was changed to open a dictionary that records the scenario id and a Trajectory list. The returned value of the previously mentioned function is appended to this list. If the list contains some value, it is stored as a json file with the scenario ID as the name. If not, the file is not stored. The controller that we want to use is also chosen in the Scenario Environment setup part of the main code.

## Result:
The JSON file recording is happening successfully. It was ensured that the JSON Control emulates the recorded data exactly by recording the videos using Keyboard Control and JSON Control, and comparing them.

# Further Work:
1. Currently, the json file that is to be accessed is being hardcoded. If it is possible to access scenario id in the ManualControlPolicy, this can be avoided. This will ensure that tthe right json file is accessed for the right scenario.
2. Hybrid Control Policy and JSON control have to be merged.
3. The recorded data has to be used for training.
