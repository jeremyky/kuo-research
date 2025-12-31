# Github is down, (no internet connection on computer)
# current working version 1:46AM OCT 4 before dynamic spring force based on speed is added

# - ALSO NEED TO REDUCE DELAY
# - ALSO NEED TO MAKE UI So user can see speed, direction, etc.
# - also add a toggle for first person
# - also, make the acceleration more realistic, rn its WAY too fast.
# add initial idle creep
# add a dynamic wheel return speed based on ego car velocity
# add more spring resistance based on ego car velocity

#!/usr/bin/env python
"""
Steering Wheel Controller using OLD constant-force autocenter logic,
implemented as a shared background thread (no races).

- Input via evdev with your exact axis mapping:
  * Steering axis 0 : already in [-1..1], 1 (left) .. -1 (right)
  * Throttle axis 2 : 1 (idle) .. -1 (full)  -> pressed in [0..1] via (1 - v)/2
  * Brake    axis 5 : 1 (idle) .. -1 (full)  -> pressed in [0..1]

- Haptics via SDL2:
  * Same centering math as your old AutoCenterWheel() (deadzone, exponent, return-to-center speed)
  * Runs in a daemon thread
  * Reads `shared.steering` and `shared.speed_kmh` written by process_input()
  * Runs the haptic effect ONCE at start; loop only updates it (no per-frame re-run)
  * Cleans up cleanly: stop/destroy effect, close haptics & joystick, ungrab evdev

- FIX: Use SDL_HAPTIC_POLAR with positive levels to avoid one-sided torque.
"""

import ctypes
import threading
import time

from metadrive.engine.core.manual_controller import SteeringWheelController




# -------- evdev ----------
try:
    import evdev
    from evdev import InputDevice, ecodes
    EVDEV_AVAILABLE = True
except Exception:
    EVDEV_AVAILABLE = False
    print("❌ evdev not available - steering wheel input will be disabled")

# -------- SDL2 ----------
SDL_AVAILABLE = True
try:
    from sdl2 import (
        SDL_Init, SDL_Quit, SDL_GetError,
        SDL_INIT_VIDEO, SDL_INIT_JOYSTICK, SDL_INIT_HAPTIC,
        SDL_JoystickOpen, SDL_NumJoysticks, SDL_JoystickName, SDL_JoystickClose,
        SDL_HapticOpenFromJoystick, SDL_Delay
    )
    from sdl2 import haptic as sdl_haptic
    from sdl2 import (
        SDL_HAPTIC_CONSTANT,
        SDL_HAPTIC_POLAR,
        SDL_HAPTIC_INFINITY,
    )
except Exception as e:
    SDL_AVAILABLE = False
    print(f"❌ SDL2 not available for haptics: {e}")


class EnhancedSteeringWheelController(SteeringWheelController):
    # ----- device mapping -----
    AXIS_STEER = 0
    AXIS_THROTTLE = 2
    AXIS_BRAKE = 5
    INVERT_PEDALS = True         # keep as you had
    INVERT_STEERING = False        # keep input sane for the sim
    # If spring pushes the wrong way even with POLAR, flip this:
    HAPTIC_SIGN_FLIP = False        # flip the spring instead of input
    FORCE_SIGN_DEADBAND = 0.005     # small deadband to prevent angle flapping at center



    STEERING_MAKEUP = 1

    # ----- idle creep -----
    IDLE_CREEP_SPEED = 0.0415
    MIN_SPEED_THRESHOLD = 0.01
    CREEP_THROTTLE = 0.15
    DRAG_COEFFICIENT = 0.15
    BRAKE_COEFFICIENT = 0.5
    VALID_INPUT_THRESHOLD = 0.95
    CREEP_ACCEL_SMOOTHING = 0.5
    CREEP_MAINTAIN_SMOOTHING = 0.9
    CREEP_SPEED_TOLERANCE = 0.005
    MAX_SPEED_KMH = 120.0

    STEERING_RATIO = 15 # car like feel
    MAX_WHEEL_DEG = 450# half travel of wheel (lock to center)
    MAX_STEER_DEG = 35 # HOw far the cars front wheels can actually turn

    INVERT_GAME_OUTPUT = False


    # for dynamic spring force autocenter at speeds (higher speed, more resistance) ; low speeds has litle autocenter like a parking lot
    speed_gain_min = .6
    speed_gain_max = 1.6
    speed_gain_v0 = 5 # lower bound for start ramp at 5 km/h
    speed_gain_v1 = 80 # reach max by 80km.h



    # ----- old autocenter opts -----
    class _Options:
        centering_deadzone = 0.0005
        centering_max_speed = 150.0
        return_to_center_multiplier = 1.8 
        centering_force_exponent = 0.3 # lower = MORE FORCE NEAR ZERO
        haptic_update_rate = 240 # previously 65. increase to try and remove DELAY
        ffb_input_factor = 1.0

    def __init__(self, enable_autocenter=True, autocenter_force=5000):
        try:
            super().__init__(enable_autocenter)
        except TypeError:
            super().__init__()

        self.autocenter_force = int(autocenter_force)
        self.options = self._Options()

        # last-known values
        self._last_steer = 0.0
        self._last_throttle = 0.0
        self._last_brake = 0.0
        self.last_throttle_brake = 0.0
        self.idle_creep_active = False

        # evdev device
        self.steering_wheel = None
        if EVDEV_AVAILABLE:
            self._load_steering_wheel()

        # shared state for haptics thread
        self._lock = threading.Lock()
        self._shared = {
            "running": False,
            "steering": 0.0,   # [-1..1], after STEERING_MAKEUP
            "speed_kmh": 0.0,
        }

        # haptics handles
        self._sdl_joystick = None
        self._haptic_device = None
        self._effect_id = -1
        self._haptics_thread = None

        if enable_autocenter and SDL_AVAILABLE:
            self._init_haptics()

    # ---------------- evdev ----------------
    def _load_steering_wheel(self):
        try:
            devices = [InputDevice(p) for p in evdev.list_devices()]
            for d in devices:
                name = (d.name or "").upper()
                if "FANATEC" in name or "WHEEL" in name or "LOGITECH" in name:
                    print(f"✅ Found wheel '{d.name}' at {d.path}")
                    try:
                        d.grab()
                        print("✅ Wheel grabbed for exclusive input")
                    except Exception as e:
                        print(f"⚠️ Could not grab wheel (continuing): {e}")
                    self.steering_wheel = d
                    print("✅ Wheel ready")
                    return
            print("❌ No steering wheel found via evdev")
        except Exception as e:
            print(f"❌ Error loading steering wheel: {e}")

    @staticmethod
    def _norm_axis_any(v):
        try:
            if -1.0 <= v <= 1.0:
                return float(v)
            if 0 <= v <= 65535:
                return (float(v) - 32767.5) / 32767.5
            if -32768 <= v <= 32767:
                return float(v) / 32767.0
        except Exception:
            pass
        return max(-1.0, min(1.0, float(v)))

    @staticmethod
    def _pressed_1_to_minus1(v):
        nv = EnhancedSteeringWheelController._norm_axis_any(v)
        return max(0.0, min(1.0, (1.0 - nv) * 0.5))

    # ---------------- haptics ----------------
    def _init_haptics(self):
        try:
            if SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC) != 0:
                print("❌ SDL_Init Error:", SDL_GetError()); return
            if SDL_NumJoysticks() < 1:
                print("❌ No joysticks for SDL haptics"); return

            js = SDL_JoystickOpen(0)
            if not js:
                print("❌ SDL_JoystickOpen Error:", SDL_GetError()); return
            self._sdl_joystick = js

            js_name = SDL_JoystickName(js).decode("utf-8") if SDL_JoystickName(js) else "unknown"
            print(f"✅ Opened Joystick for haptics: {js_name}")

            hdev = SDL_HapticOpenFromJoystick(js)
            if not hdev:
                print("❌ SDL_HapticOpenFromJoystick Error:", SDL_GetError()); return

            if not (sdl_haptic.SDL_HapticQuery(hdev) & SDL_HAPTIC_CONSTANT):
                print("❌ Haptic device does not support constant force."); return

            try:
                sdl_haptic.SDL_HapticSetGain(hdev, 100)
            except Exception:
                pass

            # Build constant effect (POLAR direction; positive levels only)
            eff = sdl_haptic.SDL_HapticEffect()
            ctypes.memset(ctypes.byref(eff), 0, ctypes.sizeof(eff))
            eff.type = SDL_HAPTIC_CONSTANT

            const = sdl_haptic.SDL_HapticConstant()
            ctypes.memset(ctypes.byref(const), 0, ctypes.sizeof(const))
            const.type = SDL_HAPTIC_CONSTANT

            direction = sdl_haptic.SDL_HapticDirection()
            ctypes.memset(ctypes.byref(direction), 0, ctypes.sizeof(direction))
            direction.type = SDL_HAPTIC_POLAR
            direction.dir[0] = 0               # start pointing 0°; we’ll update each tick
            const.direction = direction
            const.length = SDL_HAPTIC_INFINITY
            const.level = 0                     # positive level only; sign via angle
            eff.constant = const

            # EFFECT ISNT RUNNING, ITS CALLED IN LOOP BUT NEVER MAKING FOREC
            effect_id = sdl_haptic.SDL_HapticNewEffect(hdev, eff)


            

            if effect_id < 0:
                print("❌ SDL_HapticNewEffect Error:", SDL_GetError()); return

            # Run ONCE; we only update effect params in the loop
            sdl_haptic.SDL_HapticRunEffect(hdev, effect_id, SDL_HAPTIC_INFINITY)

            self._haptic_device = hdev
            self._effect_id = effect_id

            self._shared["running"] = True
            self._haptics_thread = threading.Thread(target=self._autocenter_loop, daemon=True)
            self._haptics_thread.start()
            print("🎯 Autocenter thread running (old logic, POLAR)")

        except Exception as e:
            print(f"❌ Failed to init autocenter: {e}")

    def _autocenter_loop(self):
        hdev = self._haptic_device
        effect_id = self._effect_id
        if not hdev or effect_id < 0:
            return

        disable_haptics_command_ = False
        last_steering_value_ = 0.0
        center_point_ = 0.0
        forward_speed_ = 0.0
        force_ = 0.0
        allow_torque_command_ = False
        additional_torque_rx_time_ = time.time()
        additional_torque_timeout_ = 0.1
        additional_torque_ = 0.0

        opt = self.options
        dt_ms = int(1000 / max(1, opt.haptic_update_rate))

        def HapticUpdate():
            nonlocal force_, center_point_, last_steering_value_, forward_speed_
            with self._lock:
                last_steering_value_ = float(self._shared["steering"])
                forward_speed_ = float(self._shared["speed_kmh"])

            if not disable_haptics_command_:
                if last_steering_value_ - opt.centering_deadzone > center_point_:
                    center_point_ = last_steering_value_ - opt.centering_deadzone
                elif last_steering_value_ + opt.centering_deadzone < center_point_:
                    center_point_ = last_steering_value_ + opt.centering_deadzone

                '''
                # this old code, rate at which the wheels center memory returns to 0 depends on the speed then slowed it down by dividing by the update rate
                so at slow speeds, slow recenter, at high speeds faster recent
                BUT this doesnt update the spring force itself, it only updated the center point offfset. it didnt pull HARDER, it just pulled back to 0 faster. 
                NEW CODE BELOW IWLL PULL FASTER AND STRONGER 
                
                if forward_speed_ >= 0.0:
                    alpha = min(forward_speed_, opt.centering_max_speed) / opt.centering_max_speed
                    alpha *= opt.return_to_center_multiplier
                    alpha /= opt.haptic_update_rate
                    alpha = min(alpha, 1.0)
                    center_point_ *= 1.0 - alpha
                    '''
                # speed away decay
                # v in km/h, aim for full effect by 80km, soft below 20
                v = max(0, min(120, forward_speed_))
                #ease in curve reaching around 1 at 80km/h swith soft threshold near 20
                t = (v - 20) / 60
                t = 0 if t < 0 else (1 if t > 1 else t)
                s = t * t
                # gentle when slow fnappier when fast, per second decay rate
                alpha_per_sec = .6 + 2.2 * s
                alpha = alpha_per_sec / opt.haptic_update_rate
                if alpha > 1:
                    alpha = 1
                center_point_ *= (1 - alpha)
                '''
                if abs(last_steering_value_) >= opt.centering_deadzone:
                    offset = last_steering_value_
                else:
                    offset = last_steering_value_ + opt.centering_deadzone
                '''
                # Deadzone shaping around absolute 0 (ignore center_point_ for now)
                import math
                offset_raw = last_steering_value_
                if abs(offset_raw) <= opt.centering_deadzone:
                    offset = 0.0
                else:
                    offset = offset_raw - math.copysign(opt.centering_deadzone, offset_raw)
                direction_sign = -1.0 if offset < 0 else 1.0
                force_ = pow(abs(offset), opt.centering_force_exponent) * direction_sign

                # new to handle dynamic autocenter / spring force at higher speeds
                ''' #STILL IMPLEMENTINGGGGGGGGGGGGGGGGGG
                v = max(0, float(forward_speed_))
                v0 = float(opt.speed_gain_v0)
                v1 = float(opt.speed_gain_v1)
                if v1 <= v0:
                    t = 1 #just incase bad configs
                else:
                    t = (v - v0) / (v1 - v0)
                    if t < 0: t = 0
                    if t > 1: t = 1
                gain = opt.speed_gain_min + (opt.speed_gain_max + opt.speed_gain_min) * t
                force_ *= gain
                '''
            else:
                force_ = 0.0

            if allow_torque_command_:
                now = time.time()
                if now <= additional_torque_rx_time_ + additional_torque_timeout_:
                    force_ += additional_torque_ * opt.ffb_input_factor

            force_clamped = max(min(force_, 1.0), -1.0)

            # --- POLAR direction: angle decides sign; level is positive magnitude ---
            # Flip sign globally if needed
            if self.HAPTIC_SIGN_FLIP:
                force_clamped = -force_clamped

            # angle = 0 if force_clamped >= 0 else 18000  # hundredths of a degree (0° or 180°)
            angle = 9000 if force_clamped >= 0 else 27000
            level = int(abs(force_clamped) * self.autocenter_force)


            level = int(abs(force_clamped) * self.autocenter_force)

            eff = sdl_haptic.SDL_HapticEffect()
            ctypes.memset(ctypes.byref(eff), 0, ctypes.sizeof(eff))
            eff.type = SDL_HAPTIC_CONSTANT
            const = sdl_haptic.SDL_HapticConstant()
            ctypes.memset(ctypes.byref(const), 0, ctypes.sizeof(const))
            const.type = SDL_HAPTIC_CONSTANT
            direction = sdl_haptic.SDL_HapticDirection()
            ctypes.memset(ctypes.byref(direction), 0, ctypes.sizeof(direction))
            direction.type = SDL_HAPTIC_POLAR
            direction.dir[0] = angle
            const.direction = direction
            const.length = SDL_HAPTIC_INFINITY
            const.level = level
            eff.constant = const

            sdl_haptic.SDL_HapticUpdateEffect(hdev, effect_id, eff)

        try:
            while True:
                with self._lock:
                    running = bool(self._shared["running"])
                if not running:
                    break
                HapticUpdate()
                SDL_Delay(dt_ms)
        except Exception as e:
            print(f"❌ Autocenter loop crashed: {e}")
        finally:
            try:
                sdl_haptic.SDL_HapticDestroyEffect(hdev, effect_id)
            except Exception:
                pass
            print("🛑 Autocenter thread stopped")

    # ---------------- main input ----------------
    def process_input(self, vehicle):


        if not self.steering_wheel:
            return [0.0, 0.0]

        steer = self._last_steer
        thr = self._last_throttle
        brk = self._last_brake

        try:
            # Drain events
            while True:
                ev = self.steering_wheel.read_one()
                if ev is None:
                    break
                if ev.type == ecodes.EV_ABS:
                    if ev.code == self.AXIS_STEER:
                        nv = self._norm_axis_any(ev.value)
                        steer = (-nv if self.INVERT_STEERING else nv)
                    elif ev.code == self.AXIS_THROTTLE:
                        thr = self._pressed_1_to_minus1(ev.value)
                    elif ev.code == self.AXIS_BRAKE:
                        brk = self._pressed_1_to_minus1(ev.value)

            # Absolute polling (prevents “stuck turn”)
            try:
                ai = self.steering_wheel.absinfo(self.AXIS_STEER)
                if ai is not None:
                    nv = self._norm_axis_any(ai.value)
                    steer = (-nv if self.INVERT_STEERING else nv)
                ai = self.steering_wheel.absinfo(self.AXIS_THROTTLE)
                if ai is not None:
                    thr = self._pressed_1_to_minus1(ai.value)
                ai = self.steering_wheel.absinfo(self.AXIS_BRAKE)
                if ai is not None:
                    brk = self._pressed_1_to_minus1(ai.value)
            except Exception:
                pass

            # Clamp
            steer = max(-1.0, min(1.0, steer))
            thr = max(0.0, min(1.0, thr))
            brk = max(0.0, min(1.0, brk))

            if self.INVERT_PEDALS:
                thr, brk = brk, thr

            self._last_steer, self._last_throttle, self._last_brake = steer, thr, brk

            # Pedals → throttle_brake
            if vehicle is not None:
                sp_norm = max(0.0, min(1.0, vehicle.speed_km_h / self.MAX_SPEED_KMH))
                no_feet = (thr < 1e-3) and (brk < 1e-3)
                if no_feet:
                    if sp_norm > self.IDLE_CREEP_SPEED:
                        drag = self.DRAG_COEFFICIENT * (-sp_norm * sp_norm)
                        throttle_brake = max(drag, -1.0)
                        self.idle_creep_active = False
                    elif abs(sp_norm - self.IDLE_CREEP_SPEED) <= self.CREEP_SPEED_TOLERANCE:
                        err = sp_norm - self.IDLE_CREEP_SPEED
                        tb = err * 0.1
                        throttle_brake = (
                            self.CREEP_MAINTAIN_SMOOTHING * self.last_throttle_brake
                            + (1.0 - self.CREEP_MAINTAIN_SMOOTHING) * tb
                        )
                        self.idle_creep_active = True
                    else:
                        target = self.CREEP_THROTTLE
                        throttle_brake = (
                            self.CREEP_ACCEL_SMOOTHING * self.last_throttle_brake
                            + (1.0 - self.CREEP_ACCEL_SMOOTHING) * target
                        )
                        self.idle_creep_active = True
                else:
                    throttle_brake = brk - thr
                    self.idle_creep_active = False
            else:
                throttle_brake = brk - thr
            '''
            throttle_brake = max(-1.0, min(1.0, throttle_brake))
            steering_cmd = max(-1.0, min(1.0, steer * self.STEERING_MAKEUP))
            self.last_throttle_brake = throttle_brake

            wheel_angle_deg = steer * MAX_WHEEL_DEG
            car_angle_deg = wheel_angle_deg / STEERING_RATIO
            steer_for_haptics = car_angle_deg / MAX_STEER_DEG
            steer_for_haptics = max(-1, min(1, steer_for_haptics))

            # Update shared for haptics
            if self._haptic_device is not None:
                with self._lock:
                    self._shared["steering"] = float(steering_cmd)
                    if vehicle is not None:
                        self._shared["speed_kmh"] = float(vehicle.speed_km_h)

            # return [steering_cmd, throttle_brake]
            steering_cmd_out = -steering_cmd
            return [steering_cmd_out, throttle_brake]
'''
            throttle_brake = max(-1, min(1, throttle_brake))
            self.last_throttle_brake = throttle_brake

            # real car ratio mapping, the steer in our normalized wheel input in [-1, ..., 1]
            wheel_angle_deg = steer * self.MAX_WHEEL_DEG
            car_angle_deg = wheel_angle_deg / self.STEERING_RATIO
            steering_game = car_angle_deg / self.MAX_STEER_DEG
            steering_game = max(-1, min(1, steering_game))
            steering_cmd_out = -steering_game if self.INVERT_GAME_OUTPUT else steering_game

            # HAPTICS INPUT: keep raw feel, not ratio mapped value
            steer_for_haptics = max(-1, min(1, steer * self.STEERING_MAKEUP))

            if self._haptic_device is not None:
                with self._lock:
                    self._shared["steering"] = float(steer_for_haptics)
                    if vehicle is not None:
                        self._shared["speed_kmh"] = float(vehicle.speed_km_h)
            return [-steering_cmd_out, throttle_brake]

        except Exception as e:
            print(f"Error reading steering wheel: {e}")
            return [0.0, 0.0]

    # ---------------- cleanup ----------------
    def cleanup(self):
        # stop thread
        try:
            with self._lock:
                self._shared["running"] = False
            if self._haptics_thread is not None:
                self._haptics_thread.join(timeout=1.0)
        except Exception:
            pass

        # stop/destroy effect and close haptics
        try:
            if self._haptic_device:
                try:
                    sdl_haptic.SDL_HapticStopAll(self._haptic_device)
                except Exception:
                    pass
                if self._effect_id is not None and self._effect_id >= 0:
                    try:
                        sdl_haptic.SDL_HapticDestroyEffect(self._haptic_device, self._effect_id)
                    except Exception:
                        pass
        finally:
            try:
                if self._haptic_device:
                    time.sleep(0.01)
                    sdl_haptic.SDL_HapticClose(self._haptic_device)
            except Exception:
                pass
            self._haptic_device = None
            self._effect_id = -1

        # close SDL joystick
        try:
            if self._sdl_joystick is not None:
                SDL_JoystickClose(self._sdl_joystick)
        except Exception:
            pass
        self._sdl_joystick = None

        # ungrab evdev so next run doesn't fight with a stale grab
        try:
            if self.steering_wheel is not None:
                try:
                    self.steering_wheel.ungrab()
                except Exception:
                    pass
            self.steering_wheel = None
        except Exception:
            pass

        # finally quit SDL
        try:
            if SDL_AVAILABLE:
                SDL_Quit()
        except Exception:
            pass


def create_enhanced_steering_wheel_controller(enable_autocenter=True, autocenter_force=2000):
    return EnhancedSteeringWheelController(
        enable_autocenter=enable_autocenter,
        autocenter_force=autocenter_force
    )



