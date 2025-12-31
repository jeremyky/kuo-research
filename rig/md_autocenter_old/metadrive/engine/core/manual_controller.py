import math

import numpy as np
from direct.controls.InputState import InputState
from .autocenter import *

from metadrive.utils import is_win, is_mac

if (not is_win()) and (not is_mac()):
    try:
        import evdev
        from evdev import ecodes, InputDevice
    except ImportError:
        pass

from metadrive.utils import import_pygame
import threading

pygame, gfxdraw = import_pygame()


class Controller:
    def process_input(self, vehicle):
        raise NotImplementedError

    def process_others(self, *args, **kwargs):
        pass


class KeyboardController(Controller):
    STEERING_INCREMENT = 0.04
    STEERING_DECAY = 0.25
    STEERING_INCREMENT_WHEN_INVERSE_DIRECTION = 0.25

    THROTTLE_INCREMENT = 0.1
    THROTTLE_DECAY = 0.2

    BRAKE_INCREMENT = 0.5
    BRAKE_DECAY = 0.5

    def __init__(self, pygame_control):
        self.pygame_control = pygame_control
        if self.pygame_control:
            pygame.init()
        else:
            self.inputs = InputState()
            self.inputs.watchWithModifiers('forward', 'w')
            self.inputs.watchWithModifiers('reverse', 's')
            self.inputs.watchWithModifiers('turnLeft', 'a')
            self.inputs.watchWithModifiers('turnRight', 'd')
            self.inputs.watchWithModifiers('takeover', 'space')
        self.steering = 0.
        self.throttle_brake = 0.
        self.takeover = False
        self.np_random = np.random.RandomState(None)

    def process_input(self, vehicle):
        if not self.pygame_control:
            left_key_pressed = right_key_pressed = up_key_pressed = down_key_pressed = False
            if self.inputs.isSet('turnLeft'):
                left_key_pressed = True
            if self.inputs.isSet('turnRight'):
                right_key_pressed = True
            if self.inputs.isSet('forward'):
                up_key_pressed = True
            if self.inputs.isSet('reverse'):
                down_key_pressed = True
            if self.inputs.isSet('takeover'):
                self.takeover = True
            else:
                self.takeover = False
        else:
            key_press = pygame.key.get_pressed()
            left_key_pressed = key_press[pygame.K_a]
            right_key_pressed = key_press[pygame.K_d]
            up_key_pressed = key_press[pygame.K_w]
            down_key_pressed = key_press[pygame.K_s]
            # TODO: We haven't implement takeover event when using Pygame renderer.

        # If no left or right is pressed, steering decays to the center.
        if not (left_key_pressed or right_key_pressed):
            if self.steering > 0.:
                self.steering -= self.STEERING_DECAY
                self.steering = max(0., self.steering)
            elif self.steering < 0.:
                self.steering += self.STEERING_DECAY
                self.steering = min(0., self.steering)
        elif left_key_pressed:
            if self.steering >= 0.0:  # If left is pressed and steering is in left, increment the steering a little bit.
                self.steering += self.STEERING_INCREMENT
            else:  # If left is pressed but steering is in right, steering back to left side a little faster.
                self.steering += self.STEERING_INCREMENT_WHEN_INVERSE_DIRECTION
        elif right_key_pressed:
            if self.steering <= 0.:  # If right is pressed and steering is in right, increment the steering a little
                self.steering -= self.STEERING_INCREMENT
            else:  # If right is pressed but steering is in left, steering back to right side a little faster.
                self.steering -= self.STEERING_INCREMENT_WHEN_INVERSE_DIRECTION

        # If no up or down is pressed, throttle decays to the center.
        if not (up_key_pressed or down_key_pressed):
            if self.throttle_brake > 0.:
                self.throttle_brake -= self.THROTTLE_DECAY
                self.throttle_brake = max(self.throttle_brake, 0.)
            elif self.throttle_brake < 0.:
                self.throttle_brake += self.BRAKE_DECAY
                self.throttle_brake = min(0., self.throttle_brake)
        elif up_key_pressed:
            self.throttle_brake = max(self.throttle_brake, 0.)
            self.throttle_brake += self.THROTTLE_INCREMENT
        elif down_key_pressed:
            self.throttle_brake = min(self.throttle_brake, 0.)
            self.throttle_brake -= self.BRAKE_INCREMENT

        rand = self.np_random.rand() / 10000
        self.steering += rand

        self.throttle_brake = min(max(-1., self.throttle_brake), 1.)
        self.steering = min(max(-1., self.steering), 1.)

        return np.array([self.steering, self.throttle_brake], dtype=np.float64)

    def process_others(self, takeover_callback=None):
        """This function allows the outer loop to call callback if some signal is received by the controller."""
        if (takeover_callback is None) or (not self.pygame_control) or (not pygame.get_init()):
            return
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                # Here we allow user to press T for takeover callback.
                takeover_callback()


class SteeringWheelController(Controller):
    RIGHT_SHIFT_PADDLE = 4
    LEFT_SHIFT_PADDLE = 5
    STEERING_MAKEUP = 1.5
    
    # Refined constants for gentler behavior
    IDLE_CREEP_SPEED = 0.0415  # Target speed for idle creep (normalized 0-1) - according to wiki idle creep speed is about 10 km/h
    MIN_SPEED_THRESHOLD = 0.01  # Speed below which idle creep activates
    CREEP_THROTTLE = 0.15  # Much gentler initial throttle for creep
    DRAG_COEFFICIENT = 0.15  # Increased drag for better slowdown
    BRAKE_COEFFICIENT = 0.5  # Increased brake force when no input
    VALID_INPUT_THRESHOLD = .95  # detect valid input for throttle/brake pedal
    CREEP_ACCEL_SMOOTHING = .5  # faster response when starting from stop
    CREEP_MAINTAIN_SMOOTHING = .9  # for smooth behavior when maintaining speed
    CREEP_SPEED_TOLERANCE = .005  # how close to target speed before switching to maintain mode

    def __init__(self, enable_autocenter=True):
        super().__init__()
        self.steering_wheel = None
        self.ffb_dev = None  # Force feedback device
        self.right_shift_paddle = False
        self.left_shift_paddle = False
        self.button_circle = False
        self.button_rectangle = False
        self.button_triangle = False
        self.button_x = False
        self.button_up = False
        self.button_down = False
        self.button_right = False
        self.button_left = False
        self.idle_creep_active = False
        self.last_throttle = 0.0
        self.auto_center_thread = None

        # First initialize evdev for input
        self._load_steering_wheel()
        
        # Then initialize SDL2 for force feedback
        if self.steering_wheel is not None and enable_autocenter:
            try:
                from sdl2 import SDL_Init, SDL_INIT_JOYSTICK, SDL_INIT_HAPTIC
                if SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC) == 0:
                    self.auto_center_thread = threading.Thread(target=AutoCenterWheel, daemon=True)
                    self.auto_center_thread.start()
                    print("Successfully initialized force feedback w/ autocenter")
                else:
                    print("Failed to initialize SDL for force feedback")
            except ImportError:
                print("SDL2 not available - force feedback will be disabled")
            except Exception as e:
                print(f"Error initializing force feedback: {str(e)}")
        elif self.steering_wheel is not None:
            print("autocenter disabled, using input only")

    def _load_steering_wheel(self):
        """Load the steering wheel device for input."""
        try:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            for device in devices:
                if "FANATEC" in device.name.upper():
                    print(f"Found Fanatec wheel at {device.path}")
                    try:
                        device.grab()
                        print("Successfully grabbed device for input")
                        self.steering_wheel = device
                        return
                    except Exception as e:
                        print(f"Failed to grab device: {str(e)}")
            print("No steering wheel found")
        except Exception as e:
            print(f"Error loading steering wheel: {str(e)}")

    def process_input(self, vehicle):
        if not self.steering_wheel:
            return [0, 0]

        try:
            # Initialize values
            steering = 0
            throttle = 1  # Default to 1 (not pressed) for pedals
            brake = 1     # Default to 1 (not pressed) for pedals
            
            # Read all pending events
            events = self.steering_wheel.read()
            for event in events:
                if event.type == evdev.ecodes.EV_ABS:
                    if event.code == 0:  # Steering axis
                        steering = -((event.value - 32767) / 32767)  # Negate for correct direction
                    elif event.code == 2:  # Gas pedal
                        throttle = event.value / 65535  # Normalize to 0-1
                    elif event.code == 5:  # Brake pedal
                        brake = event.value / 65535  # Normalize to 0-1
                elif event.type == evdev.ecodes.EV_KEY:
                    if event.code == self.RIGHT_SHIFT_PADDLE:
                        self.right_shift_paddle = bool(event.value)
                    elif event.code == self.LEFT_SHIFT_PADDLE:
                        self.left_shift_paddle = bool(event.value)

            print(f"\n=== Input Debug ===")
            print(f"Raw pedal values - Throttle: {throttle:.3f}, Brake: {brake:.3f}")

            if vehicle is not None:
                current_speed = vehicle.speed_km_h / 120.0  # Normalize speed to 0-1 range
                print(f"Current speed: {vehicle.speed_km_h:.1f} km/h (normalized: {current_speed:.3f})")
                
                if throttle > self.VALID_INPUT_THRESHOLD and brake > self.VALID_INPUT_THRESHOLD:  # No valid pedal input
                    print("State: No pedal input detected")
                    
                    if current_speed > self.IDLE_CREEP_SPEED:
                        print("Behavior: Car moving - applying stronger slowdown")
                        drag_force = self.DRAG_COEFFICIENT * (-current_speed * current_speed)
                        throttle_brake = max(drag_force, -1)
                        print(f"  - Drag force: {drag_force:.3f}")
                        print(f"  - Throttle brake chosen value: {throttle_brake:.3f}")
                        self.idle_creep_active = False
                    
                    elif abs(current_speed - self.IDLE_CREEP_SPEED) <= self.CREEP_SPEED_TOLERANCE:
                        print("Behavior: In creep speed range - maintaining")
                        speed_error = current_speed - self.IDLE_CREEP_SPEED
                        throttle_brake = speed_error * .1
                        throttle_brake = self.CREEP_MAINTAIN_SMOOTHING * self.last_throttle + (1 - self.CREEP_MAINTAIN_SMOOTHING) * throttle_brake
                        self.idle_creep_active = True
                        
                    else:  # Below creep speed
                        print("Behavior: Below creep speed - accelerating")
                        target_throttle = self.CREEP_THROTTLE
                        throttle_brake = self.CREEP_ACCEL_SMOOTHING * self.last_throttle + (1 - self.CREEP_ACCEL_SMOOTHING) * target_throttle
                        self.idle_creep_active = True
                else:
                    print("State: Active pedal input")
                    # Normal pedal input: brake is positive, throttle is negative
                    throttle_brake = brake - throttle
                    self.idle_creep_active = False
            else:
                print("State: No vehicle data available")
                throttle_brake = brake - throttle

            # Clamp final values
            throttle_brake = max(min(throttle_brake, 1.0), -1.0)
            steering = max(min(steering * self.STEERING_MAKEUP, 1.0), -1.0)
            
            print(f"Final values - Steering: {steering:.3f}, Throttle/Brake: {throttle_brake:.3f}")
            print(f"Idle creep active: {self.idle_creep_active}")
            print("================\n")

            self.last_throttle = throttle_brake
            return [steering, throttle_brake]

        except BlockingIOError:
            # No events available, return last values
            return [0, self.last_throttle]
        except Exception as e:
            print(f"Error reading steering wheel: {str(e)}")
            return [0, 0]

    def process_others(self, takeover_callback=None):
        """Process other inputs like shift paddles."""
        if takeover_callback and (self.left_shift_paddle or self.right_shift_paddle):
            takeover_callback()


class XboxController(Controller):
    """Control class for Xbox wireless controller
    Accept both wired and wireless connection
    Max steering, throttle, and break are bound by _discount.

    See https://www.pygame.org/docs/ref/joystick.html#xbox-360-controller-pygame-2-x for key mapping.
    """
    STEERING_DISCOUNT = 0.5
    THROTTLE_DISCOUNT = 0.4
    BREAK_DISCOUNT = 0.5

    BUTTON_A_MAP = 0
    BUTTON_B_MAP = 1
    BUTTON_X_MAP = 2
    BUTTON_Y_MAP = 3

    STEERING_AXIS = 0  # Left stick left-right direction.
    THROTTLE_AXIS = 3  # Right stick up-down direction.
    TAKEOVER_AXIS_2 = 4  # Right trigger
    TAKEOVER_AXIS_1 = 5  # Left trigger

    def __init__(self):
        try:
            import evdev
            from evdev import ecodes, InputDevice
        except ImportError:
            print(
                "Fail to load evdev, which is required for steering wheel control. "
                "Install evdev via pip install evdev"
            )
        pygame.display.init()
        pygame.joystick.init()
        assert not is_win(), "Joystick is supported in linux and mac only"
        assert pygame.joystick.get_count() > 0, "Please connect joystick or use keyboard input"
        print("Successfully Connect your Joystick!")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.button_x = False
        self.button_y = False
        self.button_a = False
        self.button_b = False

        self.button_up = False
        self.button_down = False
        self.button_right = False
        self.button_left = False

    def process_input(self, vehicle):
        pygame.event.pump()
        steering = -self.joystick.get_axis(self.STEERING_AXIS)
        if abs(steering) < 0.05:
            steering = 0
        elif steering < 0:
            steering = -(math.pow(2, abs(steering) * self.STEERING_DISCOUNT) - 1)
        else:
            steering = math.pow(2, abs(steering) * self.STEERING_DISCOUNT) - 1

        raw_throttle_brake = -self.joystick.get_axis(self.THROTTLE_AXIS)
        if abs(raw_throttle_brake) < 0.05:
            throttle_brake = 0
        elif raw_throttle_brake < 0:
            throttle_brake = -(math.pow(2, abs(raw_throttle_brake) * self.BREAK_DISCOUNT) - 1)
        else:
            throttle_brake = math.pow(2, abs(raw_throttle_brake) * self.THROTTLE_DISCOUNT) - 1

        self.takeover = (
            self.joystick.get_axis(self.TAKEOVER_AXIS_2) > -0.9 or self.joystick.get_axis(self.TAKEOVER_AXIS_1) > -0.9
        )

        self.button_x = True if self.joystick.get_button(self.BUTTON_X_MAP) else False
        self.button_y = True if self.joystick.get_button(self.BUTTON_Y_MAP) else False
        self.button_a = True if self.joystick.get_button(self.BUTTON_A_MAP) else False
        self.button_b = True if self.joystick.get_button(self.BUTTON_B_MAP) else False

        hat = self.joystick.get_hat(0)
        self.button_up = True if hat[-1] == 1 else False
        self.button_down = True if hat[-1] == -1 else False
        self.button_left = True if hat[0] == -1 else False
        self.button_right = True if hat[0] == 1 else False

        return [steering, throttle_brake]