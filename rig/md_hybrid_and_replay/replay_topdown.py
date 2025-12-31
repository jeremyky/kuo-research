# replay_topdown.py
#!/usr/bin/env python3
import argparse, os, time
from datetime import datetime
import pygame as pg

from metadrive.envs.custom_scenario_env import CustomScenarioEnv

# --- optional gif playback deps
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False
from PIL import Image, ImageSequence

# -------- Colors & Legend --------
EGO_RGB    = (247, 181,   0)  # yellow
OTHER_RGB  = (  0,   0,   0)  # black
PED_RGB    = ( 46, 204, 113)  # green
OBJ_RGB    = (127, 140, 141)  # gray

LEGEND_ITEMS_BASE = [
    ("Ego vehicle",       EGO_RGB),
    ("Other vehicles",    OTHER_RGB),
    ("Pedestrians",       PED_RGB),
    ("Traffic objects",   OBJ_RGB),
]
LEGEND_EGO_TRAJ = ("Ego trajectory (red)", (220, 20, 60))

HELP_LINES = [
    "[H] help  [L] legend  [M] semantic  [C] center-map  [F] follow-ego  [T] full traj",
    "[ [ / ] ] history len   [+/-] zoom (window size)   [,/.] speed 0.25x..4x   [SPACE] pause  [S] step",
    "[R] rec  [G] save GIF  [P] play GIF   [PgUp/PgDn] next/prev scenario   [N] restart SAME   [Q]/Esc quit"
]

SPEED_STEPS = [0.25, 0.5, 1.0, 2.0, 4.0]

# ---------------- helpers ----------------
def sanitize_for_env(raw_cfg: dict) -> dict:
    default_cfg = CustomScenarioEnv.default_config().get_dict()
    allowed = set(default_cfg.keys())
    clean = {k: v for k, v in raw_cfg.items() if k in allowed}
    for k in ("agent_configs", "sensors"):
        if k in default_cfg and k not in clean:
            clean[k] = default_cfg[k]
    clean.update(dict(use_render=False, log_level=50))
    return clean

def load_config(path):
    import json
    with open(path, "r") as f:
        return sanitize_for_env(json.load(f))

def gif_name(out_dir=".", prefix="traj"):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{prefix}_{ts}.gif")

def play_gif_external(path):
    if not os.path.exists(path):
        print(f"[GIF] not found: {path}")
        return
    print(f"[GIF] playing {path} (press q to close)")
    if HAS_CV2:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            while True:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok: break
                cv2.imshow("GIF Playback", frame)
                if cv2.waitKey(25) & 0xFF in (ord('q'), 27): break
            cap.release(); cv2.destroyAllWindows(); return
        else:
            print("[GIF] OpenCV failed to open; falling back to Pillow.")
    try:
        _ = [f.copy() for f in ImageSequence.Iterator(Image.open(path))]
        print("[GIF] Loaded with Pillow; install OpenCV to get a playback window.")
    except Exception:
        print("[GIF] could not play")

def get_current_scenario_handle(env):
    dm = getattr(env, "engine", None)
    dm = getattr(dm, "data_manager", None) if dm else None
    sid, sidx = None, None
    if dm:
        for attr in ("current_scenario_id", "curr_scenario_id"):
            if hasattr(dm, attr) and isinstance(getattr(dm, attr), str):
                sid = getattr(dm, attr); break
        if sid is None and hasattr(dm, "current_scenario"):
            val = getattr(dm, "current_scenario")
            if isinstance(val, dict) and "id" in val: sid = val["id"]
        for attr in ("current_scenario_index", "_start_index", "start_index"):
            if hasattr(dm, attr):
                try: sidx = int(getattr(dm, attr)); break
                except Exception: pass
    if sidx is None and hasattr(env, "current_seed"):
        try: sidx = int(env.current_seed)
        except Exception: pass
    return sidx, sid

def infer_total_scenarios(env, base_cfg):
    dm = getattr(env, "engine", None)
    dm = getattr(dm, "data_manager", None) if dm else None
    for attr in ("total_scenarios", "_total_scenarios", "num_scenarios", "scenario_num"):
        if dm and hasattr(dm, attr):
            try:
                val = int(getattr(dm, attr))
                if val > 0: return val
            except Exception:
                pass
    for attr in ("all_scenarios", "scenarios", "_scenarios", "scenario_indices"):
        if dm and hasattr(dm, attr):
            try:
                return len(getattr(dm, attr))
            except Exception:
                pass
    if "num_scenarios" in base_cfg and isinstance(base_cfg["num_scenarios"], int) and base_cfg["num_scenarios"] > 0:
        return base_cfg["num_scenarios"]
    return None

def build_cfg_for_scenario(base_cfg, sidx):
    cfg = dict(base_cfg)
    cfg["start_scenario_index"] = int(sidx)
    cfg["num_scenarios"] = 1
    cfg["sequential_seed"] = False
    cfg["only_reset_when_replay"] = False
    return cfg

# ---- force colors (fixes “all cars are yellow” when semantic map is off)
def force_vehicle_colors(env, ego_rgb=EGO_RGB, other_rgb=OTHER_RGB):
    try:
        ego = getattr(env, "vehicle", None) or getattr(env, "agent_vehicle", None)
        if ego is not None:
            if hasattr(ego, "top_down_color"): ego.top_down_color = ego_rgb
            elif hasattr(ego, "COLOR"):        ego.COLOR = ego_rgb
        tm = getattr(getattr(env, "engine", None), "traffic_manager", None)
        if not tm: return
        vehicles_container = None
        for name in ("vehicles", "traffic_vehicles", "_vehicles"):
            if hasattr(tm, name):
                vehicles_container = getattr(tm, name); break
        if vehicles_container is None: return
        if isinstance(vehicles_container, dict):
            vehicles = list(vehicles_container.values())
        else:
            vehicles = list(vehicles_container)
        for v in vehicles:
            if v is ego: continue
            if hasattr(v, "top_down_color"): v.top_down_color = other_rgb
            elif hasattr(v, "COLOR"):        v.COLOR = other_rgb
    except Exception:
        pass

# --- robust episode-end detector
def is_episode_done(step_ret, info, env, steps_taken):
    # Gymnasium: (obs, reward, terminated, truncated, info)
    if isinstance(step_ret, tuple):
        if len(step_ret) == 5:
            _, _, terminated, truncated, _ = step_ret
            if bool(terminated) or bool(truncated):
                return True
        elif len(step_ret) == 4:
            # Gym: (obs, reward, done, info)
            _, _, done, _ = step_ret
            if bool(done):
                return True
    # info-based hints (MetaDrive variants)
    if isinstance(info, dict):
        flags = (
            "done", "episode_done", "replay_done", "should_end", "time_out",
            "arrive_destination", "out_of_route_done", "crash_vehicle_done",
            "crash_object_done", "crash_human_done", "max_step_reached", "success"
        )
        if any(info.get(k, False) for k in flags):
            return True
    # env attributes seen in the wild
    for attr in ("done", "episode_done", "terminated", "truncated"):
        if hasattr(env, attr) and bool(getattr(env, attr)):
            return True
    # horizon fallback
    horizon = None
    try:
        horizon = int(getattr(env, "horizon", None) or env.config.get("horizon", None))
    except Exception:
        pass
    if horizon and steps_taken >= max(1, horizon - 1):
        return True
    return False

# ---------------- pygame overlay (cached) ----------------
class Overlay:
    def __init__(self, screen_size):
        if not pg.font.get_init():
            pg.font.init()
        self.font = pg.font.SysFont("arial", 14)
        self._legend_on = True
        self._help_on = True
        self._legend_surface = None
        self._help_surface = None
        self.center_reticle = False
        self._rebuild_help()
        self._rebuild_legend(LEGEND_ITEMS_BASE, show_traj=False)

    def set_help(self, on):
        if on != self._help_on:
            self._help_on = on
            self._rebuild_help()

    def set_legend(self, on):
        if on != self._legend_on:
            self._legend_on = on
            self._rebuild_legend(LEGEND_ITEMS_BASE, show_traj=False)

    def set_legend_traj(self, show_traj):
        self._rebuild_legend(LEGEND_ITEMS_BASE, show_traj=show_traj)

    def set_center_reticle(self, on): self.center_reticle = on

    def _panel(self, w, h, alpha=190):
        s = pg.Surface((w, h), pg.SRCALPHA).convert_alpha()
        s.fill((255, 255, 255, alpha))
        return s

    def _rebuild_help(self):
        if not self._help_on:
            self._help_surface = None; return
        line_h = 18; pad = 8
        width = 940
        height = pad*2 + line_h*len(HELP_LINES)
        s = self._panel(width, height)
        y = pad
        for ln in HELP_LINES:
            txt = self.font.render(ln, True, (0,0,0))
            s.blit(txt, (pad, y)); y += line_h
        self._help_surface = s

    def _rebuild_legend(self, legend_items, show_traj=False):
        if not self._legend_on:
            self._legend_surface = None; return
        items = list(legend_items)
        if show_traj: items.append(LEGEND_EGO_TRAJ)
        line_h = 18; pad = 8
        width = 360
        height = pad*2 + line_h*len(items)
        s = self._panel(width, height)
        y = pad
        for name, rgb in items:
            pg.draw.rect(s, rgb, pg.Rect(pad, y+2, 18, 12))
            txt = self.font.render(name, True, (0,0,0))
            s.blit(txt, (pad+26, y)); y += line_h
        self._legend_surface = s

    def _draw_center_reticle(self, screen):
        if not self.center_reticle: return
        cx, cy = screen.get_width() // 2, screen.get_height() // 2
        pg.draw.circle(screen, (255, 0, 0), (cx, cy), 10, 2)
        pg.draw.circle(screen, (255, 255, 255), (cx, cy), 16, 2)
        pg.draw.line(screen, (255, 0, 0), (cx-12, cy), (cx-2, cy), 2)
        pg.draw.line(screen, (255, 0, 0), (cx+2, cy), (cx+12, cy), 2)
        pg.draw.line(screen, (255, 0, 0), (cx, cy-12), (cx, cy-2), 2)
        pg.draw.line(screen, (255, 0, 0), (cx, cy+2), (cx, cy+12), 2)
        label = self.font.render("EGO", True, (255, 255, 255))
        screen.blit(label, (cx + 14, cy - 8))

    def draw(self, screen, status_text):
        if self._help_surface is not None:
            screen.blit(self._help_surface, (8, 8))
        if self._legend_surface is not None:
            y = 8 + (self._help_surface.get_height() + 8 if self._help_surface else 0)
            screen.blit(self._legend_surface, (8, y))
        self._draw_center_reticle(screen)
        txt = self.font.render(status_text, True, (20,20,20))
        screen.blit(txt, (8, screen.get_height()-24))

# ---------------- main ----------------
def main():
    # ensure we’re not in headless SDL
    for k in ("SDL_VIDEODRIVER", "SDL_AUDIODRIVER"):
        if os.environ.get(k, "") == "dummy":
            del os.environ[k]

    ap = argparse.ArgumentParser()
    # Semantic default ON with override flags
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--semantic", dest="semantic", action="store_true")
    g.add_argument("--no-semantic", dest="semantic", action="store_false")
    ap.set_defaults(semantic=True)

    ap.add_argument("--config", required=True)
    ap.add_argument("--center_on_map", action="store_true")
    ap.add_argument("--num_stack", type=int, default=60)
    ap.add_argument("--history_smooth", type=int, default=0)
    ap.add_argument("--draw_full_traj", action="store_true", default=True)  # default ON
    ap.add_argument("--scaling", type=int, default=5)          # keep scaling fixed (zoom via window size)
    ap.add_argument("--film", type=int, nargs=2, default=[2000,2000])
    ap.add_argument("--screen", type=int, nargs=2, default=[1000,1000])
    ap.add_argument("--gif_dir", type=str, default=".")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--auto_restart_sec", type=float, default=5.0)  # auto replay delay
    args = ap.parse_args()

    base_cfg = load_config(args.config)

    # Probe once for scenario count/id
    probe = CustomScenarioEnv(base_cfg); probe.reset()
    sidx0, sid0 = get_current_scenario_handle(probe)
    scenario_total = infer_total_scenarios(probe, base_cfg)
    probe.close()

    # UI defaults: semantic ON, follow ON, full trajectory ON
    ui = dict(
        semantic=bool(args.semantic),
        center=False,          # center-map OFF (since follow is ON)
        follow=True,           # follow ego ON
        full_traj=bool(args.draw_full_traj),
        num_stack=args.num_stack,
        history_smooth=args.history_smooth,
        legend=True, help=True,
        recording=False, last_gif=None,
        quit=False, restart=False,
        zoom=1.0,
        speed_index=SPEED_STEPS.index(1.0),
        paused=False, step_once=False,
        scenario_index=int(sidx0) if sidx0 is not None else 0,
        scenario_total=int(scenario_total) if scenario_total else None,
        scenario_id=str(sid0) if sid0 else None,
        auto_restart_sec=float(args.auto_restart_sec),
    )

    base_screen = tuple(args.screen)

    pg.init(); clock = pg.time.Clock()

    print("\nHotkeys: " + " | ".join([
        "H help", "L legend", "M semantic", "C center-map", "F follow-ego", "T full traj",
        "[/ ] history", "+/- zoom", ",/. speed", "SPACE pause", "S step",
        "PgUp next", "PgDn prev", "N restart (same)", "R rec", "G save", "P play", "Q/Esc quit"
    ]))

    def make_env():
        cfg = build_cfg_for_scenario(base_cfg, ui["scenario_index"])
        return CustomScenarioEnv(cfg)

    def ensure_window(env_local, screen_size):
        env_local.render(mode="topdown", window=True, screen_size=screen_size)
        surf = pg.display.get_surface()
        if surf is None or surf.get_size() != screen_size:
            pg.display.set_mode(screen_size)
            surf = pg.display.get_surface()
        return surf

    def dynamic_screen():
        w, h = base_screen
        z = max(0.5, min(4.0, ui["zoom"]))
        sw = max(320, min(4096, int(w / z)))
        sh = max(240, min(4096, int(h / z)))
        return (sw, sh)

    def wrap_next_index(delta):
        if ui["scenario_total"] and ui["scenario_total"] > 0:
            ui["scenario_index"] = (ui["scenario_index"] + delta) % ui["scenario_total"]
        else:
            ui["scenario_index"] = max(0, ui["scenario_index"] + delta)

    def draw_countdown(seconds_left):
        screen = pg.display.get_surface()
        if not screen: return
        txt = f"Auto-restart in {max(0,int(seconds_left)+1)}s  [N restart now | Q/Esc cancel]"
        font = pg.font.SysFont("arial", 16)
        surf = font.render(txt, True, (0,0,0))
        pad = 8
        panel = pg.Surface((surf.get_width()+pad*2, surf.get_height()+pad*2), pg.SRCALPHA)
        panel.fill((255,255,255,210))
        x = screen.get_width() - panel.get_width() - 10
        y = 10
        screen.blit(panel, (x, y))
        screen.blit(surf, (x+pad, y+pad))
        pg.display.flip()

    def post_episode_countdown(seconds):
        deadline = time.time() + seconds
        while time.time() < deadline and not ui["quit"] and not ui["restart"]:
            for ev in pg.event.get():
                if ev.type == pg.QUIT: ui["quit"] = True
                elif ev.type == pg.KEYDOWN:
                    if ev.key in (pg.K_ESCAPE, pg.K_q): ui["quit"] = True
                    elif ev.key == pg.K_n: ui["restart"] = True
            draw_countdown(deadline - time.time())
            clock.tick(20)
        if not ui["quit"] and not ui["restart"]:
            ui["restart"] = True

    def episode_loop():
        env_local = make_env()
        obs, _ = env_local.reset()

        # capture id/index for HUD
        sidx_now, sid_now = get_current_scenario_handle(env_local)
        if sidx_now is not None: ui["scenario_index"] = int(sidx_now)
        if sid_now: ui["scenario_id"] = str(sid_now)

        # apply color fix immediately (only when not semantic)
        if not ui["semantic"]:
            force_vehicle_colors(env_local, EGO_RGB, OTHER_RGB)

        screen = ensure_window(env_local, dynamic_screen())

        if not pg.font.get_init(): pg.font.init()
        overlay = Overlay(screen.get_size())
        overlay.set_help(ui["help"]); overlay.set_legend(ui["legend"])
        overlay.set_legend_traj(ui["full_traj"]); overlay.set_center_reticle(ui["follow"])

        running = True
        last_screen_size = screen.get_size()

        # episode step counter (fallback for horizon)
        steps_taken = 0
        episode_done = False

        while running and (not ui["quit"]) and (not ui["restart"]):
            for ev in pg.event.get():
                if ev.type == pg.QUIT: ui["quit"] = True
                elif ev.type == pg.KEYDOWN:
                    k = ev.key
                    if k in (pg.K_ESCAPE, pg.K_q): ui["quit"] = True
                    elif k == pg.K_h: ui["help"] = not ui["help"]; overlay.set_help(ui["help"])
                    elif k == pg.K_l: ui["legend"] = not ui["legend"]; overlay.set_legend(ui["legend"])
                    elif k == pg.K_m: ui["semantic"] = not ui["semantic"]
                    elif k == pg.K_c:
                        ui["center"] = not ui["center"]
                        if ui["center"]:
                            ui["follow"] = False; overlay.set_center_reticle(False)
                    elif k == pg.K_f:
                        ui["follow"] = not ui["follow"]
                        if ui["follow"]: ui["center"] = False
                        overlay.set_center_reticle(ui["follow"])
                    elif k == pg.K_t:
                        ui["full_traj"] = not ui["full_traj"]; overlay.set_legend_traj(ui["full_traj"])
                    elif k == pg.K_RIGHTBRACKET: ui["num_stack"] = min(ui["num_stack"]+10, 2000)
                    elif k == pg.K_LEFTBRACKET:  ui["num_stack"] = max(ui["num_stack"]-10, 0)
                    elif k in (pg.K_PLUS, pg.K_EQUALS):  ui["zoom"] = min(ui["zoom"] * 1.15, 4.0)
                    elif k == pg.K_MINUS:               ui["zoom"] = max(ui["zoom"] / 1.15, 0.5)
                    elif k == pg.K_COMMA:               ui["speed_index"] = max(0, ui["speed_index"] - 1)
                    elif k == pg.K_PERIOD:              ui["speed_index"] = min(len(SPEED_STEPS)-1, ui["speed_index"] + 1)
                    elif k == pg.K_SPACE:               ui["paused"] = not ui["paused"]
                    elif k == pg.K_s:
                        if ui["paused"]: ui["step_once"] = True
                    elif k == pg.K_PAGEUP:              wrap_next_index(+1); ui["restart"] = True; running = False
                    elif k == pg.K_PAGEDOWN:            wrap_next_index(-1); ui["restart"] = True; running = False
                    elif k == pg.K_n:                   ui["restart"] = True; running = False
                    elif k == pg.K_r:
                        ui["recording"] = not ui["recording"]; print(f"[REC] {'ON' if ui['recording'] else 'OFF'}")
                    elif k == pg.K_g:
                        if hasattr(env_local, "top_down_renderer") and getattr(env_local.top_down_renderer, "_frames", None):
                            out = gif_name(args.gif_dir, "traj")
                            env_local.top_down_renderer.generate_gif(file_name=out, fps=20)
                            ui["last_gif"] = out; print(f"[GIF] saved {out}")
                        else:
                            print("[GIF] no frames recorded; press R to record first.")
                    elif k == pg.K_p:
                        if ui["last_gif"]: play_gif_external(ui["last_gif"])
                        else: print("[GIF] nothing to play yet (save with G)")

            # adapt window to zoom
            desired = dynamic_screen()
            if desired != last_screen_size:
                screen = ensure_window(env_local, desired)
                overlay = Overlay(screen.get_size())
                overlay.set_help(ui["help"]); overlay.set_legend(ui["legend"])
                overlay.set_legend_traj(ui["full_traj"]); overlay.set_center_reticle(ui["follow"])
                last_screen_size = desired

            # speed + pause/step
            speed = SPEED_STEPS[ui["speed_index"]]
            should_step = True
            steps_this_frame = 1
            if ui["paused"]:
                should_step = ui["step_once"]; ui["step_once"] = False
            else:
                if speed > 1.0: steps_this_frame = int(speed)

            # step env and detect episode end robustly
            if should_step and not episode_done:
                for _ in range(steps_this_frame):
                    try:
                        step_out = env_local.step(env_local.agent_policy([obs]) if hasattr(env_local, "agent_policy") else [0.0, 0.0])
                    except Exception:
                        # If stepping raises, treat as end (common with exhausted replays)
                        episode_done = True
                        break
                    info = step_out[-1] if isinstance(step_out, tuple) else {}
                    episode_done = is_episode_done(step_out, info, env_local, steps_taken)
                    # unpack obs for next step
                    if isinstance(step_out, tuple):
                        obs = step_out[0]
                    steps_taken += 1
                    if episode_done: break

            # enforce proper colors each frame (only when not semantic)
            if not ui["semantic"]:
                force_vehicle_colors(env_local, EGO_RGB, OTHER_RGB)

            # render (even if paused or episode_done)
            env_local.render(
                mode="topdown",
                window=True,
                screen_size=last_screen_size,
                scaling=args.scaling,                   # scaling fixed; window size changes instead
                film_size=tuple(args.film),
                num_stack=ui["num_stack"],
                history_smooth=ui["history_smooth"],
                semantic_map=ui["semantic"],
                draw_target_vehicle_trajectory=ui["full_traj"],   # ego traj visible
                center_on_map=ui["center"],
                target_agent_heading_up=ui["follow"],             # follow-ego default ON
                screen_record=ui["recording"],
            )

            # HUD
            scen_disp = f"{ui['scenario_index']}" + (f"/{ui['scenario_total']}" if ui["scenario_total"] else "")
            sid_disp  = f" id={ui['scenario_id']}" if ui["scenario_id"] else ""
            status = (
                f"scenario={scen_disp}{sid_disp}  zoom={ui['zoom']:.2f}x  speed={speed}x  "
                f"paused={ui['paused']}  semantic={ui['semantic']}  center-map={ui['center']}  "
                f"follow-ego={ui['follow']}  full_traj={ui['full_traj']}  num_stack={ui['num_stack']}  "
                f"rec={ui['recording']}"
            )
            overlay.draw(screen, status)
            pg.display.flip()

            # frame pacing: slow-mo lowers FPS; fast-forward keeps base fps
            target_fps = args.fps * (speed if (speed < 1.0 and not ui['paused']) else 1.0)
            target_fps = max(5, int(target_fps))
            clock.tick(target_fps)

            if episode_done:
                running = False

        # keep the last frame visible; then countdown+restart if requested
        if not ui["quit"]:
            if ui["recording"]:
                ui["recording"] = False
                out = gif_name(args.gif_dir, "traj_autosave")
                try:
                    env_local.top_down_renderer.generate_gif(file_name=out, fps=20)
                    ui["last_gif"] = out; print(f"[GIF] autosaved {out}")
                except Exception:
                    pass
            if not ui["restart"] and episode_done and ui["auto_restart_sec"] > 0:
                post_episode_countdown(ui["auto_restart_sec"])

        env_local.close()
        time.sleep(0.1)

    try:
        while not ui["quit"]:
            ui["restart"] = False
            episode_loop()
            if not ui["restart"]:
                break
    finally:
        pg.quit()

if __name__ == "__main__":
    main()



