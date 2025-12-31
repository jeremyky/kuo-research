import sys
import ctypes
import time
from sdl2 import *
from sdl2 import haptic

class Options:
    centering_deadzone = 0.002
    centering_max_speed = 100.0
    return_to_center_multiplier = 1.0
    centering_force_exponent = 0.5
    haptic_update_rate = 60
    ffb_input_factor = 1.0

def AutoCenterWheel():
    # Initialize SDL
    if SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC) != 0:
        print("SDL_Init Error:", SDL_GetError())
        return

    # Open the joystick
    num_joysticks = SDL_NumJoysticks()
    if num_joysticks < 1:
        print("No joysticks connected!")
        SDL_Quit()
        return
    
    joystick = SDL_JoystickOpen(0)
    joystick_name = SDL_JoystickName(joystick).decode('utf-8')
    if "pedal" in joystick_name.lower():
        joystick = SDL_JoystickOpen(1)

    if not joystick:
        print("SDL_JoystickOpen Error:", SDL_GetError())
        SDL_Quit()
        return

    joystick_name = SDL_JoystickName(joystick).decode('utf-8')
    print("Opened Joystick:", joystick_name)

    # Initialize haptic device
    haptic_device = SDL_HapticOpenFromJoystick(joystick)
    if not haptic_device:
        print("SDL_HapticOpenFromJoystick Error:", SDL_GetError())
        SDL_JoystickClose(joystick)
        SDL_Quit()
        return

    if not (SDL_HapticQuery(haptic_device) & SDL_HAPTIC_CONSTANT):
        print("Haptic device does not support constant force.")
        SDL_HapticClose(haptic_device)
        SDL_JoystickClose(joystick)
        SDL_Quit()
        return

    # Define the haptic effect
    effect = haptic.SDL_HapticEffect()
    ctypes.memset(ctypes.byref(effect), 0, ctypes.sizeof(effect))
    effect.type = SDL_HAPTIC_CONSTANT

    constant = haptic.SDL_HapticConstant()
    ctypes.memset(ctypes.byref(constant), 0, ctypes.sizeof(constant))
    constant.type = SDL_HAPTIC_CONSTANT

    direction = haptic.SDL_HapticDirection()
    ctypes.memset(ctypes.byref(direction), 0, ctypes.sizeof(direction))
    direction.type = SDL_HAPTIC_CARTESIAN
    direction.dir[0] = 0
    direction.dir[1] = 0
    constant.direction = direction

    constant.length = SDL_HAPTIC_INFINITY
    constant.level = 0
    effect.constant = constant

    # Create the effect
    effect_id = SDL_HapticNewEffect(haptic_device, effect)
    if effect_id < 0:
        print("SDL_HapticNewEffect Error:", SDL_GetError())
        SDL_HapticClose(haptic_device)
        SDL_JoystickClose(joystick)
        SDL_Quit()
        return

    # Variables and options
    disable_haptics_command_ = False
    last_steering_value_ = 0.0
    center_point_ = 0.0
    forward_speed_ = 0.0
    force_ = 0.0
    allow_torque_command_ = False
    additional_torque_rx_time_ = time.time()
    additional_torque_timeout_ = 0.1  # 100 milliseconds
    additional_torque_ = 0.0

    options_ = Options()

    # Helper function to update haptic feedback
    def HapticUpdate():
        nonlocal force_, center_point_, effect, additional_torque_, allow_torque_command_

        if not disable_haptics_command_:
            # Adjust center point
            if last_steering_value_ - options_.centering_deadzone > center_point_:
                center_point_ = last_steering_value_ - options_.centering_deadzone
            elif last_steering_value_ + options_.centering_deadzone < center_point_:
                center_point_ = last_steering_value_ + options_.centering_deadzone

            # Calculate return-to-center speed
            if forward_speed_ >= 0.0:
                alpha = min(forward_speed_, options_.centering_max_speed) / options_.centering_max_speed
                alpha *= options_.return_to_center_multiplier
                alpha /= options_.haptic_update_rate
                alpha = min(alpha, 1.0)
                center_point_ *= 1.0 - alpha

            # Calculate centering force
            offset_from_centering_point = last_steering_value_ if abs(last_steering_value_) >= options_.centering_deadzone else last_steering_value_ + options_.centering_deadzone
            direction = -1.0 if offset_from_centering_point < 0 else 1.0
            force_ = pow(abs(offset_from_centering_point), options_.centering_force_exponent) * direction

        else:
            force_ = 0.0

        # Handle additional torque commands
        if allow_torque_command_:
            current_time = time.time()
            if current_time <= additional_torque_rx_time_ + additional_torque_timeout_:
                force_ += additional_torque_ * options_.ffb_input_factor

        # Clamp force value and update the haptic effect
        force_ = max(min(force_, 1.0), -1.0)
        effect.constant.level = int(force_ * 15000)

        if haptic_device:
            SDL_HapticUpdateEffect(haptic_device, effect_id, effect)
            SDL_HapticRunEffect(haptic_device, effect_id, 1)

    # Main loop
    running = True
    event = SDL_Event() #maybe it detects both pedal and wheel events??
    while running:
        while SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == SDL_QUIT:
                running = False
            elif event.type == SDL_JOYAXISMOTION:
                # For example, axis 0 is steering
                if event.jaxis.axis == 0:
                    last_steering_value_ = event.jaxis.value / 32767.0  # Normalize between -1 and 1
                elif event.jaxis.axis == 1:
                    forward_speed_ = (event.jaxis.value / 32767.0) * options_.centering_max_speed

        # Update haptic feedback
        HapticUpdate()

        # Delay to control update rate
        SDL_Delay(int(1000 / options_.haptic_update_rate))

    # Cleanup
    SDL_HapticDestroyEffect(haptic_device, effect_id)
    SDL_HapticClose(haptic_device)
    SDL_JoystickClose(joystick)
    SDL_Quit()

# Example usage
if __name__ == "__main__":
    AutoCenterWheel()