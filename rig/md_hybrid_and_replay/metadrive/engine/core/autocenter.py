# metadrive/engine/core/autocenter.py
import ctypes
from sdl2 import *
from sdl2 import haptic


"""
Auto-centering force feedback for steering wheels using SDL2.


Key fixes vs. older versions:
- Proper constant-force direction (X axis) with sign carried in `level`
- Initial center calibration window, then slow bias adaption
- Intent-aware reduction so it doesn't fight you while you are turning
- Gentle damping to avoid oscillation, without blocking turns
- Start the haptic effect once (infinite) and only update its level
"""


class Options:
    # Strength cap mapped to SDL's signed 16-bit force range (±32767).
    # 3000–7000 is a gentle/medium range for most consumer wheels.
    centering_force_max = 5000


    # Deadzone around center before any centering is applied (in wheel units [-1..1])
    centering_deadzone = 0.01


    # How aggressively force grows with error (0.4–0.8 is soft; >1 is snappier)
    centering_force_exponent = 0.6


    # Max speed (km/h) used to normalize speed-based scaling
    centering_max_speed = 100.0


    # Update rate (Hz) for the haptics loop
    haptic_update_rate = 60


    # How much to reduce centering when user is actively turning (0..1)
    # Effective reduction is 0.6 * turn_intent by default
    intent_reduction = 0.6


    # Initial calibration window (ms) to average the wheel position as "center"
    initial_calibration_ms = 500


    # Slow bias adaptation factor when near center (prevents long-term drift)
    center_adapt_alpha = 0.02  # smaller = slower drift




def _clamp_int16(x: int) -> int:
    return max(min(x, 32767), -32768)




def AutoCenterWheel(shared_data: dict) -> None:
    """
    Background thread function.


    Expected shared_data keys (all optional but recommended):
      - 'haptic_device': SDL_Haptic* (required)
      - 'steering_value': float in [-1, 1]
      - 'speed': float (km/h)
      - 'running': bool
    """
    print("[FFB] Autocenter thread started (SDL2)")


    # --- Validate device ---
    haptic_device = shared_data.get("haptic_device")
    if not haptic_device:
        print("[FFB] No haptic device in shared_data; autocenter disabled")
        return


    if not (SDL_HapticQuery(haptic_device) & SDL_HAPTIC_CONSTANT):
        print("[FFB] Device does not support SDL_HAPTIC_CONSTANT; autocenter disabled")
        return


    # --- Build the constant-force effect (direction on X axis) ---
    effect = haptic.SDL_HapticEffect()
    ctypes.memset(ctypes.byref(effect), 0, ctypes.sizeof(effect))
    effect.type = SDL_HAPTIC_CONSTANT


    const = haptic.SDL_HapticConstant()
    ctypes.memset(ctypes.byref(const), 0, ctypes.sizeof(const))
    const.type = SDL_HAPTIC_CONSTANT


    direction = haptic.SDL_HapticDirection()
    ctypes.memset(ctypes.byref(direction), 0, ctypes.sizeof(direction))
    direction.type = SDL_HAPTIC_CARTESIAN
    direction.dir[0] = 1   # X axis (left/right)
    direction.dir[1] = 0
    direction.dir[2] = 0
    const.direction = direction


    const.length = SDL_HAPTIC_INFINITY  # keep running indefinitely
    const.level = 0                     # start neutral
    effect.constant = const


    effect_id = SDL_HapticNewEffect(haptic_device, effect)
    if effect_id < 0:
        print("[FFB] SDL_HapticNewEffect error:", SDL_GetError())
        return


    # Start once; we will just update the effect params each tick
    if SDL_HapticRunEffect(haptic_device, effect_id, 1) < 0:
        print("[FFB] SDL_HapticRunEffect error:", SDL_GetError())
        SDL_HapticDestroyEffect(haptic_device, effect_id)
        return


    # --- Control variables ---
    opts = Options()
    hz = max(20, int(opts.haptic_update_rate))
    tick_ms = int(1000 / hz)


    center_point = 0.0
    prev_val = 0.0


    # Initial center calibration (average over a short window)
    t0 = SDL_GetTicks()
    accum = 0.0
    count = 0


    print("[FFB] Calibrating center... (hands off wheel if possible)")
    try:
        while shared_data.get("running", True):
            # Pull current inputs (default safe values)
            val = float(shared_data.get("steering_value", 0.0))  # [-1, 1]
            speed = float(shared_data.get("speed", 0.0))         # km/h


            now = SDL_GetTicks()


            # --- Initial center calibration window ---
            if now - t0 < opts.initial_calibration_ms:
                accum += val
                count += 1
                center_point = accum / max(count, 1)
            else:
                # Gentle bias adaptation only when near center, so it won't yank
                if abs(val - center_point) < 0.2:  # "near center" heuristic
                    center_point = (1.0 - opts.center_adapt_alpha) * center_point + \
                                   opts.center_adapt_alpha * val


            # --- Error & deadzone ---
            error = val - center_point
            if abs(error) < opts.centering_deadzone:
                error = 0.0


            # --- Base centering (opposes error), soft-shaped by exponent ---
            if error != 0.0:
                base = - (abs(error) ** opts.centering_force_exponent) * (1.0 if error > 0 else -1.0)
            else:
                base = 0.0


            # --- Speed scaling (gentler when stopped) ---
            speed_factor = min(max(speed / max(opts.centering_max_speed, 1e-6), 0.0), 1.0)
            base *= (0.4 + 0.6 * speed_factor)  # 40% at standstill → 100% at speed


            # --- Damping (prevents oscillation but doesn't block turning) ---
            dv = val - prev_val
            prev_val = val
            damping = -0.15 * dv  # small, symmetric


            # --- Intent-aware reduction (don’t fight the user while turning) ---
            turn_intent = min(abs(dv) * hz / 2.0, 1.0)  # rough rate → [0..1]
            intent_gain = 1.0 - opts.intent_reduction * turn_intent
            intent_gain = max(0.0, min(1.0, intent_gain))


            # --- Combine and clamp ---
            raw_force = (base + damping) * intent_gain
            raw_force = max(min(raw_force, 1.0), -1.0)


            # --- Scale to device units and push update ---
            level = int(opts.centering_force_max * raw_force)
            effect.constant.level = _clamp_int16(level)


            if SDL_HapticUpdateEffect(haptic_device, effect_id, effect) < 0:
                # If update fails, attempt to continue next tick
                err = SDL_GetError()
                # Avoid spamming logs every tick
                # print("[FFB] SDL_HapticUpdateEffect error:", err)
                pass


            SDL_Delay(tick_ms)


    finally:
        SDL_HapticStopEffect(haptic_device, effect_id)
        SDL_HapticDestroyEffect(haptic_device, effect_id)
        print("[FFB] Autocenter thread stopped")





