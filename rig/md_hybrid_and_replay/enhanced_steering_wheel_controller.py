#!/usr/bin/env python
"""
Enhanced (lean) steering wheel controller for MetaDrive.

Goals:
- Deterministic lock/unlock/clamp flow across 4 stages in data collection.
- "Global/middle zero": baseline center is the unwrapped 0.0 (not a near alias).
- Constant-force spring toward global zero while locked.
- Minimal shaping; robust to device hiccups (effect recreation, sign auto-cal).

Public API used by the app:
- hard_reset_sync(device_reset_time_s=0.0, center_lock_s=..., spring_boost=...)
- reset_autocenter()                     # baseline = global middle zero
- set_spring(enabled: bool)              # start/stop haptics spring safely
- lock_input(locked: bool)               # neutralize outputs while locked
- set_clamp(enabled: bool)               # NEW: hard clamp (vs push-to-center)
- process_input(vehicle_or_None) -> [steering, throttle_brake]
- is_centered(deadband_deg=2.0, settle_s=0.4) -> bool
- get_center_error() -> degrees
- auto_calibrate_sign(duration_s=0.25, poll_hz=100)
- autocenter_force (int) attribute

If SDL2 haptics are unavailable, spring calls are safe no-ops and readiness can
fall back to angle-only dwell.
"""

import ctypes
import threading
import time
import numpy as np

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
        SDL_JoystickOpen, SDL_JoystickClose, SDL_NumJoysticks, SDL_JoystickName,
        SDL_Delay
    )
    from sdl2 import haptic as sdl_haptic
    from sdl2 import SDL_HAPTIC_CONSTANT, SDL_HAPTIC_POLAR, SDL_HAPTIC_INFINITY
except Exception as e:
    SDL_AVAILABLE = False
    print(f"❌ SDL2 not available for haptics: {e}")


class EnhancedSteeringWheelController(SteeringWheelController):
    # ---- device axes (adjust if needed) ----
    AXIS_STEER = 0
    AXIS_THROTTLE = 2
    AXIS_BRAKE = 5

    # ---- mapping toggles ----
    INVERT_STEERING = False       # input sign flip
    INVERT_PEDALS = True          # swap throttle/brake after normalization
    INVERT_GAME_OUTPUT = False    # extra flip on final steering to game

    # ---- wheel/vehicle geometry ----
    MAX_WHEEL_DEG = 450.0         # half-travel (= 900° total)
    STEERING_RATIO = 15.0         # wheel:road
    MAX_STEER_DEG = 35.0          # vehicle rack limit

    # ---- simple pedal shaping ----
    THR_DEADZONE = 0.01
    BRK_DEADZONE = 0.02

    # ---- spring parameters ----
    autocenter_force = 3000       # externally tunable (int)
    _spring_gain = 1.0            # internal multiplier
    _center_lock_deadline = 0.0   # when startup boost ends
    _start_boost_gain = 1.0
    HAPTIC_SIGN_FLIP = False      # flips torque globally if device sign is inverted

    # ---- readiness detector ----
    _settle_deadband_deg = 2.0
    _settle_time_s = 0.40

    def __init__(self, enable_autocenter=True, autocenter_force=3000):
        try:
            super().__init__(enable_autocenter)
        except TypeError:
            super().__init__()

        # runtime flags
        self._enable_autocenter_flag = bool(enable_autocenter)
        self.autocenter_force = int(autocenter_force)
        self._input_locked = False          # neutralize outputs while locked
        self._clamp_enabled = False         # NEW: hard clamp flag (managed by pipeline)
        self._i_accum = 0.0                 # small integral nudge for push-to-center

        # evdev device
        self.steering_wheel = None
        if EVDEV_AVAILABLE:
            self._load_steering_wheel()

        # haptics shared state
        self._lock = threading.Lock()
        self._shared = {"running": False, "steering": 0.0, "speed_kmh": 0.0}

        # SDL handles
        self._sdl_joystick = None
        self._haptic_device = None
        self._effect_id = -1
        self._haptics_thread = None

        # unwrapped angle / global zero
        self._last_raw_for_unwrap = None
        self._revolutions = 0.0              # increments by ±1.0 per wrap across ±1.0
        self._angle_unwrapped_norm = 0.0     # raw_norm + 2*k
        self._baseline_unwrapped_norm = 0.0  # GLOBAL target zero (middle zero)
        self._center_error_deg = 0.0

        # readiness
        self._ready_event = threading.Event()
        self._settle_since = None

        if self._enable_autocenter_flag and SDL_AVAILABLE:
            self._init_haptics()

    # ----------------------------------------------------------------------
    # Public lifecycle
    # ----------------------------------------------------------------------
    def hard_reset_sync(self, device_reset_time_s=0.0, center_lock_s=1.0, spring_boost=1.5):
        """
        Close old inputs/haptics; reopen clean; arm a lock window with stronger spring.
        TARGET is raw 0.0 (manual's zero). Initialize unwrapped angle from hardware
        so the spring physically drives to manual's center.

        Gate entry should call:
          set_spring(True), lock_input(True), set_clamp(False)  # push-to-center
        """
        self._ready_event.clear()
        self._settle_since = None
        self._i_accum = 0.0

        self.cleanup()
        if device_reset_time_s > 0:
            time.sleep(device_reset_time_s)

        if EVDEV_AVAILABLE:
            self._load_steering_wheel()
        if self._enable_autocenter_flag and SDL_AVAILABLE:
            self._init_haptics()

        # Initialize UNWRAPPED angle to CURRENT hardware reading (post INVERT_STEERING).
        # Baseline stays at 0.0 → target == manual zero.
        self._revolutions = 0.0
        cur = float(self._read_current_normalized_steer())
        self._last_raw_for_unwrap = cur
        self._angle_unwrapped_norm = cur
        self._baseline_unwrapped_norm = 0.0
        self._center_error_deg = (self._angle_unwrapped_norm - self._baseline_unwrapped_norm) * self.MAX_WHEEL_DEG

        # Gate starts in PUSH mode (clamp disabled).
        self._clamp_enabled = False

        # arm lock window (startup boost)
        self._center_lock_deadline = time.time() + max(0.0, float(center_lock_s))
        self._start_boost_gain = float(max(1.0, spring_boost))
        self.resume_autocenter(0.0)

    def reset_autocenter(self):
        """Re-baseline global zero at current unwrapped angle and clear readiness."""
        self._baseline_unwrapped_norm = float(self._angle_unwrapped_norm)
        self._settle_since = None
        self._ready_event.clear()

    def set_spring(self, enabled: bool):
        if enabled:
            self.resume_autocenter(0.0)
        else:
            self.pause_autocenter()

    def lock_input(self, locked: bool):
        if locked:
            self.lock_inputs()
        else:
            self.unlock_inputs()

    def lock_inputs(self):
        self._input_locked = True
        print("[Wheel] lock_inputs(True)")

    def unlock_inputs(self):
        self._input_locked = False
        print("[Wheel] lock_inputs(False)")

    # NEW: independent clamp switch (pipeline toggles this per phase)
    def set_clamp(self, enabled: bool):
        """Enable/disable hard clamp (independent of input locking)."""
        self._clamp_enabled = bool(enabled)
        print(f"[Wheel] set_clamp({self._clamp_enabled})")

    def is_centered(self, deadband_deg: float = None, settle_time_s: float = None) -> bool:
        """
        True when |center_error| <= deadband for >= settle_time.
        If the haptics thread is running, the thread maintains _ready_event.
        Otherwise, fall back to computing dwell here from the latest _center_error_deg.
        """
        if deadband_deg is not None:
            self._settle_deadband_deg = float(deadband_deg)
        if settle_time_s is not None:
            self._settle_time_s = float(settle_time_s)

        # If haptics thread is alive, let it be authoritative.
        haptics_alive = bool(self._haptics_thread is not None)
        if haptics_alive:
            return self._ready_event.is_set()

        # Angle-only fallback: compute dwell readiness here.
        in_band = abs(float(self._center_error_deg)) <= float(self._settle_deadband_deg)
        now = time.time()
        if in_band:
            self._settle_since = self._settle_since or now
            if (now - (self._settle_since or now)) >= float(self._settle_time_s):
                return True
        else:
            self._settle_since = None
        return False

    def get_center_error(self) -> float:
        return float(self._center_error_deg)

    def auto_calibrate_sign(self, duration_s=0.25, poll_hz=100):
        """
        Try to detect inverted torque sign: while spring is active and trying to center,
        if |center_error| consistently grows, flip HAPTIC_SIGN_FLIP.
        """
        if not SDL_AVAILABLE or not self._haptic_device:
            return
        dt = 1.0 / max(1, int(poll_hz))
        t_end = time.time() + max(0.05, float(duration_s))
        # sample trend
        last = abs(self._center_error_deg)
        worse_count, better_count = 0, 0
        while time.time() < t_end:
            # haptics loop updates _center_error_deg; we just observe
            cur = abs(self._center_error_deg)
            if cur > last + 0.25:  # threshold in degrees to ignore noise
                worse_count += 1
            elif cur < last - 0.25:
                better_count += 1
            last = cur
            SDL_Delay(int(dt * 1000))
        if worse_count > better_count:
            self.HAPTIC_SIGN_FLIP = not self.HAPTIC_SIGN_FLIP
            print(f"[Wheel] Auto-cal: flipping torque sign → HAPTIC_SIGN_FLIP={self.HAPTIC_SIGN_FLIP}")

    # ----------------------------------------------------------------------
    # Haptics lifecycle
    # ----------------------------------------------------------------------
    def pause_autocenter(self):
        try:
            with self._lock:
                self._shared["running"] = False
            if self._haptics_thread:
                self._haptics_thread.join(timeout=0.5)
                self._haptics_thread = None
            if self._haptic_device:
                try:
                    sdl_haptic.SDL_HapticStopAll(self._haptic_device)
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ pause_autocenter failed: {e}")

    def resume_autocenter(self, delay_s: float = 0.0):
        try:
            if delay_s > 0:
                time.sleep(delay_s)
            if not self._haptic_device:
                # SDL might be unavailable; spring is a no-op in that case.
                return
            if self._effect_id is None or self._effect_id < 0:
                self._effect_id = self._new_constant_effect(self._haptic_device)

            # run (recreate on failure)
            try:
                sdl_haptic.SDL_HapticRunEffect(self._haptic_device, self._effect_id, SDL_HAPTIC_INFINITY)
            except Exception:
                try:
                    if self._effect_id is not None and self._effect_id >= 0:
                        sdl_haptic.SDL_HapticDestroyEffect(self._haptic_device, self._effect_id)
                except Exception:
                    pass
                self._effect_id = self._new_constant_effect(self._haptic_device)
                sdl_haptic.SDL_HapticRunEffect(self._haptic_device, self._effect_id, SDL_HAPTIC_INFINITY)

            with self._lock:
                self._shared["running"] = True
            if self._haptics_thread is None:
                self._haptics_thread = threading.Thread(target=self._autocenter_loop, daemon=True)
                self._haptics_thread.start()
        except Exception as e:
            print(f"⚠️ resume_autocenter failed: {e}")

    # ----------------------------------------------------------------------
    # Input processing
    # ----------------------------------------------------------------------
    def process_input(self, vehicle):
        """
        Returns [steering, throttle_brake] where +tb = brake, -tb = throttle.
        While locked, returns [0,0] but still feeds haptics with current steering.
        """
        if not self.steering_wheel:
            # still feed spring with last known value (already in _shared)
            return [0.0, 0.0]

        steer = 0.0
        thr = 0.0
        brk = 0.0

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

            # Absolute polling (avoids stale axes)
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

            # clip and (optionally) swap pedals
            steer = max(-1.0, min(1.0, steer))
            thr = max(0.0, min(1.0, thr))
            brk = max(0.0, min(1.0, brk))
            if self.INVERT_PEDALS:
                thr, brk = brk, thr

            # ---------- unwrap to compute GLOBAL center error ----------
            raw_now = float(steer)
            if self._last_raw_for_unwrap is not None:
                delta = raw_now - self._last_raw_for_unwrap
                # detect wrap across ±1.0 → adjust by ±2.0
                if delta < -1.5:
                    self._revolutions += 1.0
                elif delta > 1.5:
                    self._revolutions -= 1.0
            self._last_raw_for_unwrap = raw_now
            self._angle_unwrapped_norm = raw_now + 2.0 * self._revolutions

            # center error in degrees relative to GLOBAL baseline
            self._center_error_deg = (self._angle_unwrapped_norm - self._baseline_unwrapped_norm) * self.MAX_WHEEL_DEG

            # ---------- feed spring loop ----------
            if self._haptic_device is not None:
                with self._lock:
                    # feed unwrapped so spring targets the middle zero across revolutions
                    self._shared["steering"] = float(self._angle_unwrapped_norm)
                    if vehicle is not None and hasattr(vehicle, "speed_km_h"):
                        self._shared["speed_kmh"] = float(vehicle.speed_km_h)

            # while locked → neutral outputs (spring fed above)
            if self._input_locked:
                return [0.0, 0.0]

            # ---------- map to game ----------
            wheel_angle_deg = raw_now * self.MAX_WHEEL_DEG
            car_angle_deg = wheel_angle_deg / self.STEERING_RATIO
            steering_game = max(-1.0, min(1.0, car_angle_deg / self.MAX_STEER_DEG))
            steering_cmd_out = -steering_game if self.INVERT_GAME_OUTPUT else steering_game

            # simple tb: +brake, -throttle
            tb = brk - thr
            tb = max(-1.0, min(1.0, tb))

            # MetaDrive expects [steer, throttle_brake]; (common sign quirk on steer)
            # Return as np.array for consistency with KeyboardController
            return np.array([(-steering_cmd_out), tb], dtype=np.float64)

        except Exception as e:
            print(f"⚠️ process_input error: {e}")
            return np.array([0.0, 0.0], dtype=np.float64)

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------
    def _game_steer_from_raw_offset(self, raw_offset_norm: float) -> float:
        """
        Convert a raw *offset* in normalized wheel units (unwrapped) into the same
        steering domain that process_input() ultimately RETURNS to MetaDrive.
        This mirrors the exact mapping pipeline used in manual takeover.
        """
        wheel_angle_deg = raw_offset_norm * self.MAX_WHEEL_DEG
        car_angle_deg   = wheel_angle_deg / self.STEERING_RATIO
        steering_game   = max(-1.0, min(1.0, car_angle_deg / self.MAX_STEER_DEG))
        returned_steer  = (steering_game if self.INVERT_GAME_OUTPUT else -steering_game)
        return returned_steer

    def _read_current_normalized_steer(self) -> float:
        """One-shot poll of current wheel axis, normalized and with INVERT_STEERING applied."""
        try:
            if self.steering_wheel:
                ai = self.steering_wheel.absinfo(self.AXIS_STEER)
                if ai is not None:
                    nv = self._norm_axis_any(ai.value)
                    return (-nv if self.INVERT_STEERING else nv)
        except Exception:
            pass
        return 0.0

    def _autocenter_loop(self):
        """Two-phase centering:
           - PUSH phase (clamp OFF): actively drives wheel to the same logical zero
             as manual takeover, with an integral 'nudge' to break stiction (Stage-1).
           - CLAMP phase (clamp ON): once centered (or commanded), hold near center
             with a strong floor so you can't pull it out (Stage-2 ghost).
           Manual takeover (Stage-3/4): pipeline disables spring & clamp.
        """
        if not self._haptic_device or self._effect_id < 0:
            return

        dt_ms = 1000 // 240  # ~240 Hz
        try:
            while True:
                # pull shared state
                with self._lock:
                    running = bool(self._shared["running"])
                    unwrapped = float(self._shared["steering"])  # unwrapped, reflects INVERT_STEERING from input path
                if not running:
                    break

                # offset in RAW wheel space (unwrapped) relative to GLOBAL baseline
                offset_norm = unwrapped - float(self._baseline_unwrapped_norm)

                # startup boost window (helps initial snap)
                lock_active = time.time() < self._center_lock_deadline
                if lock_active and abs(offset_norm) < 0.02:  # ~3.6°
                    offset_norm = 0.0
                boost = self._start_boost_gain if lock_active else 1.0

                # map the *offset* through the exact manual mapping pipeline
                wheel_angle_deg = offset_norm * self.MAX_WHEEL_DEG
                car_angle_deg   = wheel_angle_deg / self.STEERING_RATIO
                steering_game   = max(-1.0, min(1.0, car_angle_deg / self.MAX_STEER_DEG))
                returned_steer  = (steering_game if self.INVERT_GAME_OUTPUT else -steering_game)

                # sign: drive returned_steer -> 0 (same convention as manual)
                if returned_steer > 1e-6:
                    direction = -1.0
                elif returned_steer < -1e-6:
                    direction = 1.0
                else:
                    direction = 0.0
                if self.HAPTIC_SIGN_FLIP:
                    direction *= -1.0

                # PUSH vs CLAMP selection (independent from input locking)
                push_mode  = not self._clamp_enabled
                clamp_mode = self._clamp_enabled

                if push_mode:
                    # --- PUSH: actively move to center (overcome static friction) ---
                    p_term = (abs(returned_steer) ** 0.35)

                    # integral nudge that grows while off-center, decays at center
                    if abs(returned_steer) > 1e-3:
                        self._i_accum = min(0.8, self._i_accum + 0.0025)
                    else:
                        self._i_accum = max(0.0, self._i_accum - 0.01)

                    # minimum helpful floor so it actually moves
                    floor_push = 0.12
                    base_mag = max(floor_push, p_term + self._i_accum)

                    # extra kick early if we're not close to center during boost window
                    if lock_active and abs(offset_norm) > 0.03:
                        base_mag = max(base_mag, 0.60)

                else:
                    # --- CLAMP: resist being pulled out; stiff near zero with a strong floor ---
                    p_term = (abs(returned_steer) ** 0.25)
                    floor_clamp = 0.45  # raise to 0.55–0.70 for "can't move it" feel
                    base_mag = max(floor_clamp, p_term)

                    # Optionally bump force if someone yanks it far off center
                    if abs(returned_steer) > 0.2:
                        base_mag = max(base_mag, 0.7)

                    # No integral accumulation while clamped
                    self._i_accum = max(0.0, self._i_accum - 0.02)

                # combine with internal gain + pre-gate boost; clamp to [0,1]
                magnitude = min(1.0, max(0.0, base_mag * self._spring_gain * boost))

                # set haptic effect: angle encodes sign (90° vs 270°), level encodes magnitude * force
                if direction == 0.0 or magnitude <= 0.0:
                    level = 0
                    angle = 9000  # arbitrary when level=0
                else:
                    angle = 9000 if direction >= 0 else 27000
                    level = int(magnitude * float(self.autocenter_force))

                self._update_constant_effect(level, angle)

                # readiness (deadband & dwell) in PHYSICAL degrees
                self._center_error_deg = offset_norm * self.MAX_WHEEL_DEG
                in_band = abs(self._center_error_deg) <= float(self._settle_deadband_deg)
                now = time.time()
                if in_band:
                    # Only accept dwell during PUSH (so clamp doesn't "auto-pass" readiness)
                    if push_mode:
                        self._settle_since = self._settle_since or now
                        if self._settle_since and (now - self._settle_since) >= float(self._settle_time_s):
                            self._ready_event.set()
                else:
                    self._settle_since = None
                    if push_mode:
                        self._ready_event.clear()

                SDL_Delay(dt_ms)
        except Exception as e:
            print(f"❌ Autocenter loop crashed: {e}")
        finally:
            try:
                sdl_haptic.SDL_HapticDestroyEffect(self._haptic_device, self._effect_id)
            except Exception:
                pass
            print("🛑 Autocenter thread stopped")

    def _update_constant_effect(self, level_int, angle_hundredths):
        try:
            eff = sdl_haptic.SDL_HapticEffect()
            ctypes.memset(ctypes.byref(eff), 0, ctypes.sizeof(eff))
            eff.type = SDL_HAPTIC_CONSTANT
            const = sdl_haptic.SDL_HapticConstant()
            ctypes.memset(ctypes.byref(const), 0, ctypes.sizeof(const))
            const.type = SDL_HAPTIC_CONSTANT
            direction = sdl_haptic.SDL_HapticDirection()
            ctypes.memset(ctypes.byref(direction), 0, ctypes.sizeof(direction))
            direction.type = SDL_HAPTIC_POLAR
            direction.dir[0] = int(angle_hundredths)
            const.direction = direction
            const.length = SDL_HAPTIC_INFINITY
            const.level = max(0, min(32767, int(level_int)))
            eff.constant = const
            sdl_haptic.SDL_HapticUpdateEffect(self._haptic_device, self._effect_id, eff)
        except Exception:
            # swallow; next loop will try again
            pass

    # ----------------------------------------------------------------------
    # Setup / teardown
    # ----------------------------------------------------------------------
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
                    return
            print("❌ No steering wheel found via evdev")
        except Exception as e:
            print(f"❌ Error loading steering wheel: {e}")

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

            name = SDL_JoystickName(js).decode("utf-8") if SDL_JoystickName(js) else "unknown"
            print(f"✅ Opened Joystick for haptics: {name}")

            hdev = sdl_haptic.SDL_HapticOpenFromJoystick(js)
            if not hdev:
                print("❌ SDL_HapticOpenFromJoystick Error:", SDL_GetError()); return

            if not (sdl_haptic.SDL_HapticQuery(hdev) & SDL_HAPTIC_CONSTANT):
                print("❌ Haptic device does not support constant force."); return

            try:
                sdl_haptic.SDL_HapticSetGain(hdev, 100)
            except Exception:
                pass

            eff_id = self._new_constant_effect(hdev)
            if eff_id < 0:
                print("❌ SDL_HapticNewEffect Error:", SDL_GetError()); return
            sdl_haptic.SDL_HapticRunEffect(hdev, eff_id, SDL_HAPTIC_INFINITY)

            self._haptic_device = hdev
            self._effect_id = eff_id

            with self._lock:
                self._shared["running"] = True
            self._haptics_thread = threading.Thread(target=self._autocenter_loop, daemon=True)
            self._haptics_thread.start()
            print("🎯 Autocenter thread running")
        except Exception as e:
            print(f"❌ Failed to init autocenter: {e}")

    @staticmethod
    def _new_constant_effect(hdev):
        eff = sdl_haptic.SDL_HapticEffect()
        ctypes.memset(ctypes.byref(eff), 0, ctypes.sizeof(eff))
        eff.type = SDL_HAPTIC_CONSTANT
        const = sdl_haptic.SDL_HapticConstant()
        ctypes.memset(ctypes.byref(const), 0, ctypes.sizeof(const))
        const.type = SDL_HAPTIC_CONSTANT
        direction = sdl_haptic.SDL_HapticDirection()
        ctypes.memset(ctypes.byref(direction), 0, ctypes.sizeof(direction))
        direction.type = SDL_HAPTIC_POLAR
        direction.dir[0] = 0  # sign is controlled by update (90° vs 270°)
        const.direction = direction
        const.length = SDL_HAPTIC_INFINITY
        const.level = 0
        eff.constant = const
        return sdl_haptic.SDL_HapticNewEffect(hdev, eff)

    def reset_between_scenarios(self, delay_s: float = 0.6):
        """Close ONLY. Next scenario re-opens in the pre-gate via hard_reset_sync()."""
        try:
            self.cleanup()
            if delay_s > 0:
                time.sleep(delay_s)
        except Exception as e:
            print(f"⚠️ reset_between_scenarios failed: {e}")

    def cleanup(self):
        # stop loop
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

        # ungrab evdev so next run doesn't fight a stale grab
        try:
            if self.steering_wheel is not None:
                try:
                    self.steering_wheel.ungrab()
                except Exception:
                    pass
            self.steering_wheel = None
        except Exception:
            pass

        # finally quit SDL (OK if nothing else needs it)
        try:
            if SDL_AVAILABLE:
                SDL_Quit()
        except Exception:
            pass

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------
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
        # maps 1(idle)..-1(full) -> 0..1 pressed
        nv = EnhancedSteeringWheelController._norm_axis_any(v)
        return max(0.0, min(1.0, (1.0 - nv) * 0.5))


def create_enhanced_steering_wheel_controller(enable_autocenter=True, autocenter_force=2000):
    return EnhancedSteeringWheelController(
        enable_autocenter=enable_autocenter,
        autocenter_force=autocenter_force
    )