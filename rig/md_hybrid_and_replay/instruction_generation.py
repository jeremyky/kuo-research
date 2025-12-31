import os
import sys
from pathlib import Path
import argparse
import json
from typing import Optional, List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

from metadrive.engine.asset_loader import AssetLoader
import metadrive.scenario.utils as sd_utils
from metadrive.scenario.scenario_description import ScenarioDescription as SD

import traceback

# =========================
# Config / Constants
# =========================

MAX_INTERSECTION_ZONES = 3
MAX_MERGE_ZONES = 5

LANE_MIN_LENGTH = 5.0         # meters
LANE_BUFFER_WIDTH = 2.5       # ~ half lane width for buffer overlap tests
INTERSECTION_MIN_AREA = 25.0  # m^2
MERGE_MIN_AREA = 15.0         # m^2

ADJ_MAX_LAT_DIST = 5.0        # candidate adjacent lane max lateral separation
ADJ_MAX_HEADING_DIFF_DEG = 15 # degrees

LATERAL_VEL_THRESH = 0.3      # m/s
LATERAL_PERSIST_SEC = 0.5     # seconds

ENTRY_LOOKBACK_STEPS = 30     # for onset_manual
EXTRA_STEPS = 10              # for extra_steps

# New lead-time requirements (so we don't instruct "inside" the zone)
MIN_TURN_LEAD_STEPS = 30          # steps before entry (≈3s at 10Hz)
MIN_TURN_LEAD_TIME_SEC = 2.0      # seconds before entry

# How many different alternatives to emit in each config
MAX_INSTRUCTION_OPTIONS = 3

# Merge temporal spacing (prevents 1-step-apart duplicates)
MERGE_MIN_STEP_GAP = 10  # steps

# Final de-dup window (applied across all instructions at the end)
FINAL_DEDUP_WINDOW = 8   # steps

# === Lateral reachability constraints for legal/feasible turns ===
LANE_WIDTH_EST = 3.6         # meters; conservative lane width
MAX_LATERAL_SPEED = 1.2     # m/s; conservative achievable lateral shift
MAX_LANES_TO_CROSS = 3     # disallow turns requiring >3 lane change
PRE_ENTRY_LOOKAHEAD_M = 8.0 # sample a cross-section ~8.0 m before intersection entry

# =========================
# Dataset path (MetaDrive)
# =========================

waymo_data = AssetLoader.file_path(AssetLoader.asset_path, "waymo", unix_style=False)


# =========================
# Top-level pipeline
# =========================

def generate(data_path: str) -> int:
    """Read scenarios, generate one config per accepted scenario, count rejects."""
    Path("configs").mkdir(parents=True, exist_ok=True)

    scenario_summaries, scenario_ids, dataset_mapping = sd_utils.read_dataset_summary(data_path)
    rejects = 0
    processed = 0
    for file_name in tqdm(scenario_summaries.keys(), desc="Generating Configs"):
        if processed % 100 == 0:
            print(f'Processed: {processed}.\nAccepted: {processed - rejects}.\nRejected: {rejects}')
        processed += 1
        try:
            abs_path = os.path.join(data_path, dataset_mapping[file_name], file_name)
            scenario = sd_utils.read_scenario_data(abs_path)
            config = config_from_scenario(scenario)
            if config:
                out_path = Path("configs") / f"{scenario[SD.ID]}.json"
                with open(out_path, "w") as f:
                    json.dump(config, f, indent=2)
            else:
                rejects += 1
        except Exception:
            # Show which file failed and the full stack trace
            print(f"[ERROR] Failed on scenario file: {file_name}")
            traceback.print_exc()
            rejects += 1

    return rejects



def config_from_scenario(scenario: SD) -> Optional[dict]:
    """Combine intersection and merge detection to build an instruction config."""
    intersections = find_intersections(scenario)
    merges = find_merges(scenario)

    if not intersections and not merges:
        return None

    return config_from_intersections_and_merges(intersections, merges, scenario)



def _dedupe_by_time_window(instructions: List[str],
                           onsets: List[int],
                           window: int) -> Tuple[List[str], List[int]]:
    """
    Keep only the first instruction that appears within 'window' steps
    of an already accepted instruction with the same text.
    """
    kept_instr = []
    kept_onsets = []
    for instr, onset in zip(instructions, onsets):
        conflict = False
        for ki, ko in zip(kept_instr, kept_onsets):
            if instr == ki and abs(onset - ko) <= window:
                conflict = True
                break
        if not conflict:
            kept_instr.append(instr)
            kept_onsets.append(onset)
    return kept_instr, kept_onsets


def config_from_intersections_and_merges(
    intersections: Optional[List[Polygon]],
    merges: Optional[List[Polygon]],
    scenario: SD
) -> Optional[dict]:
    """
    Build a multi-instruction config. Prefer intersections (with adequate lead),
    then top up with merges to reach MAX_INSTRUCTION_OPTIONS.
    """
    all_instructions: List[str] = []
    all_onsets: List[int] = []

    # 1) Try intersections first
    if intersections:
        instr, onsets = _intersection_candidates_multi(
            scenario, intersections, MAX_INSTRUCTION_OPTIONS
        )
        all_instructions += instr
        all_onsets += onsets

    # 2) If we still want more options, top up with merges
    remaining = max(0, MAX_INSTRUCTION_OPTIONS - len(all_instructions))
    if remaining and merges:
        merge_entry_info = get_merge_entry_steps(scenario, merges)
        merge_opts = select_merge_candidates(scenario, merges, merge_entry_info, remaining)
        for m in merge_opts:
            all_instructions.append(f"merge {m['direction']}")
            all_onsets.append(max(0, m["entry_step"] - ENTRY_LOOKBACK_STEPS))

    if not all_instructions:
        return None

    # 3) Final de-dup pass: avoid near-duplicate identical instructions too close in time
    all_instructions, all_onsets = _dedupe_by_time_window(
        all_instructions, all_onsets, FINAL_DEDUP_WINDOW
    )

    # 4) Clip to MAX_INSTRUCTION_OPTIONS (after de-dup)
    if len(all_instructions) > MAX_INSTRUCTION_OPTIONS:
        all_instructions = all_instructions[:MAX_INSTRUCTION_OPTIONS]
        all_onsets = all_onsets[:MAX_INSTRUCTION_OPTIONS]

    if not all_instructions:
        return None

    # Backward-compat single fields (first option)
    return {
        "instructions": all_instructions,              # list
        "onset_manuals": all_onsets,                   # list (index-aligned)
        "onset_manual": all_onsets[0],                 # legacy single
        "instructions_legacy": all_instructions[0],    # legacy single
        "reactive_traffic": True,
        "extra_steps": EXTRA_STEPS
    }


# =========================
# Lead-time filtering
# =========================

def _filter_entries_with_lead(scenario: SD,
                              entries: List[Tuple[int, int, float]],
                              min_steps: int,
                              min_time: float) -> List[Tuple[int, int, float]]:
    """Keep only (zone_id, step_idx, ts) with enough pre-entry lead."""
    ts = np.asarray(scenario["metadata"]["ts"], dtype=float)
    if len(ts) == 0:
        return []
    t0 = float(ts[0])
    out = []
    for (zid, step, ts_entry) in entries:
        ok_steps = step >= min_steps
        ok_time = (float(ts_entry) - t0) >= min_time
        if ok_steps and ok_time:
            out.append((zid, step, ts_entry))
    return out


# =========================
# Intersection helpers
# =========================

def get_intersection_entry_steps(scenario: SD, intersections: List[Polygon]) -> List[Tuple[int, int, float]]:
    """Return (intersection_id, step_idx, timestamp) when ego enters an intersection."""
    positions = scenario["tracks"][scenario["metadata"]["sdc_id"]]["state"]["position"]
    timestamps = scenario["metadata"]["ts"]
    entry_info: List[Tuple[int, int, float]] = []

    for i, poly in enumerate(intersections):
        for step_idx, pos in enumerate(positions):
            if poly.contains(Point(pos[:2])):
                entry_info.append((i, step_idx, float(timestamps[step_idx])))
                break
    return entry_info


def classify_replay_turn(positions: np.ndarray, entry: int, exit_idx: int) -> str:
    """Left / right / straight via heading change between entry and exit windows."""
    def safe_dir(a, b) -> np.ndarray:
        v = np.array(b[:2]) - np.array(a[:2])
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([1.0, 0.0])

    pre = entry - 2 if entry >= 2 else entry - 1
    pre = max(pre, 0)
    entry_dir = safe_dir(positions[pre], positions[entry])
    exit_dir = safe_dir(positions[exit_idx], positions[min(exit_idx + 2, len(positions) - 1)])

    cross = np.cross(entry_dir, exit_dir)
    dot = np.dot(entry_dir, exit_dir)
    angle_deg = np.degrees(np.arctan2(cross, dot))

    if angle_deg > 30:
        return "left"
    if angle_deg < -30:
        return "right"
    return "straight"


def get_available_turns(ego_pos: np.ndarray, ego_heading: np.ndarray, lane_lines: Dict[str, LineString]) -> List[str]:
    """Estimate feasible turn directions based on nearby lane headings."""
    candidates: List[str] = []
    for line in lane_lines.values():
        start = np.array(line.coords[0])
        if np.linalg.norm(start - ego_pos) < 10.0:
            lane_dir = np.array(line.coords[-1]) - start
            if np.linalg.norm(lane_dir) < 1e-6:
                continue
            lane_dir = lane_dir / np.linalg.norm(lane_dir)
            angle = np.degrees(np.arctan2(np.cross(ego_heading, lane_dir), np.dot(ego_heading, lane_dir)))
            turn = "left" if angle > 30 else "right" if angle < -30 else "straight"
            candidates.append(turn)
    return list(set(candidates))

# === NEW: helpers for feasibility filtering ===

def _classify_turn_of_lane_at_zone(lane: LineString, ego_heading_unit: np.ndarray) -> str:
    """Roughly classify lane's exit direction (left/right/straight) vs ego heading."""
    start = np.array(lane.coords[0], dtype=float)
    end   = np.array(lane.coords[-1], dtype=float)
    v = end - start
    n = np.linalg.norm(v)
    if n < 1e-6:
        return "straight"
    lane_dir = v / n
    cross = np.cross(ego_heading_unit, lane_dir)
    dot   = np.dot(ego_heading_unit, lane_dir)
    angle_deg = float(np.degrees(np.arctan2(cross, dot)))
    if angle_deg > 30:  return "left"
    if angle_deg < -30: return "right"
    return "straight"


def _lanes_supporting_direction_in_zone(
    lane_lines: Dict[str, LineString],
    zone: Polygon,
    ego_heading_unit: np.ndarray
) -> Dict[str, List[LineString]]:
    """Return {'left': [...], 'right': [...], 'straight': [...]} lanes that overlap the zone."""
    out = {"left": [], "right": [], "straight": []}
    for ln in lane_lines.values():
        if not ln.buffer(1.5).intersects(zone):
            continue
        t = _classify_turn_of_lane_at_zone(ln, ego_heading_unit)
        out[t].append(ln)
    return out


def _lateral_gap_to_lane_at_cross_section(
    ego_lane: LineString,
    target_lane: LineString,
    cross_xy: np.ndarray
) -> float:
    """Positive gap -> target to ego's left; negative -> to the right. Returns meters."""
    _, _, d_ego = _project_point_to_line(cross_xy, ego_lane)
    _, _, d_tgt = _project_point_to_line(cross_xy, target_lane)
    return float(d_tgt - d_ego)


def _pick_target_turn_lane(
    desired: str,
    ego_lane: LineString,
    cross_xy: np.ndarray,
    lanes_by_dir: Dict[str, List[LineString]]
) -> Optional[LineString]:
    cands = lanes_by_dir.get(desired, [])
    best = None
    best_abs_gap = 1e9
    for ln in cands:
        gap = _lateral_gap_to_lane_at_cross_section(ego_lane, ln, cross_xy)
        # side constraint
        if desired == "right" and gap >= 0:
            continue
        if desired == "left" and gap <= 0:
            continue
        ag = abs(gap)
        if ag < best_abs_gap:
            best_abs_gap = ag
            best = ln
    return best


def _is_turn_feasible(
    scenario: SD,
    entry_step: int,
    zone: Polygon,
    desired: str,
    lane_lines: Dict[str, LineString],
    ego_lane_at_entry: LineString
) -> bool:
    """Looser feasibility: small grace meters and pass if EITHER lane-count OR time test looks ok."""
    ts = np.asarray(scenario["metadata"]["ts"], dtype=float)
    if len(ts) < 2:
        return False
    dt = float(np.median(np.diff(ts)))

    # Previously: max(MIN_TURN_LEAD_STEPS*dt, MIN_TURN_LEAD_TIME_SEC)
    # Loosen: use a weighted blend that tends to the time threshold sooner.
    available_time = 0.5 * (MIN_TURN_LEAD_STEPS * dt) + 0.5 * MIN_TURN_LEAD_TIME_SEC

    positions = np.asarray(scenario["tracks"][scenario["metadata"]["sdc_id"]]["state"]["position"], dtype=float)
    pre = max(entry_step - 2, 0)
    ego_dir = positions[entry_step, :2] - positions[pre, :2]
    n = np.linalg.norm(ego_dir)
    ego_dir = ego_dir / n if n >= 1e-3 else np.array([1.0, 0.0])

    lanes_by_dir = _lanes_supporting_direction_in_zone(lane_lines, zone, ego_dir)

    ego_entry_xy = positions[entry_step, :2].astype(float)
    proj_xy, s_along, _ = _project_point_to_line(ego_entry_xy, ego_lane_at_entry)
    s_cross = max(0.0, s_along - PRE_ENTRY_LOOKAHEAD_M)
    cross_xy = np.array(ego_lane_at_entry.interpolate(s_cross).coords[0], dtype=float) if ego_lane_at_entry.length > 0.0 else proj_xy

    tgt_lane = _pick_target_turn_lane(desired if desired in ("left", "right") else "straight",
                                      ego_lane_at_entry, cross_xy, lanes_by_dir)
    if tgt_lane is None:
        return False

    # Loosen with a small grace to not punish tiny offsets
    GRACE_M = 1.2  # ~ one third of a lane
    gap_m = max(0.0, abs(_lateral_gap_to_lane_at_cross_section(ego_lane_at_entry, tgt_lane, cross_xy)) - GRACE_M)

    lanes_to_cross = gap_m / LANE_WIDTH_EST
    needed_time = gap_m / max(MAX_LATERAL_SPEED, 1e-3)

    # Softer decision: pass if (lane count ok) OR (time ok), instead of requiring both.
    lane_ok = (lanes_to_cross <= MAX_LANES_TO_CROSS + 1e-6)
    time_ok = (needed_time <= 1.25 * available_time)  # 25% slack

    return lane_ok or time_ok



def choose_different_instruction(
    scenario: SD,
    intersections: List[Polygon],
    entry_info: List[Tuple[int, int, float]],
    lane_lines: Dict[str, LineString]
) -> List[Dict]:
    """Generate counterfactual turn instructions that differ from replay, but only if feasible."""
    positions = scenario["tracks"][scenario["metadata"]["sdc_id"]]["state"]["position"]
    positions = np.asarray(positions)

    lane_list = list(lane_lines.values())

    def nearest_lane_to_point(xy: np.ndarray) -> LineString:
        return min(lane_list, key=lambda ln: ln.distance(Point(xy)))

    instructions: List[Dict] = []
    for i, entry_step, _ in entry_info:
        ego_pos = np.array(positions[entry_step][:2])
        prev_pos = np.array(positions[max(entry_step - 2, 0)][:2])
        ego_heading = ego_pos - prev_pos
        n = np.linalg.norm(ego_heading)
        if n < 1e-3:
            ego_heading = np.array([1.0, 0.0], dtype=float)
        else:
            ego_heading /= n

        exit_step = min(entry_step + 10, len(positions) - 1)
        actual_turn = classify_replay_turn(positions, entry_step, exit_step)
        available = get_available_turns(ego_pos, ego_heading, lane_lines)
        alternatives = [t for t in available if t != actual_turn]
        if not alternatives:
            continue

        # Feasibility check
        zone = intersections[i]
        ego_lane_at_entry = nearest_lane_to_point(ego_pos)

        feasible = []
        for alt in alternatives:
            desired = alt if alt in ("left", "right") else "straight"
            if _is_turn_feasible(scenario, entry_step, zone, desired, lane_lines, ego_lane_at_entry):
                feasible.append(alt)

        if not feasible:
            continue

        new_instruction = str(np.random.choice(feasible))
        instructions.append({
            "intersection_id": i,
            "entry_step": entry_step,
            "original_turn": actual_turn,
            "new_instruction": new_instruction
        })
    return instructions


def _intersection_candidates_multi(
    scenario: SD,
    intersections: List[Polygon],
    max_k: int
) -> Tuple[List[str], List[int]]:
    """
    Build up to max_k intersection-based options with adequate lead time.
    """
    entry_info = get_intersection_entry_steps(scenario, intersections)
    entry_info = _filter_entries_with_lead(scenario, entry_info,
                                           MIN_TURN_LEAD_STEPS,
                                           MIN_TURN_LEAD_TIME_SEC)
    if not entry_info:
        return [], []

    entry_info.sort()  # by intersection_id then time
    lane_lines = build_lane_lines(scenario)
    raw = choose_different_instruction(scenario, intersections, entry_info, lane_lines)

    if not raw:
        return [], []

    # Deduplicate by (intersection_id, entry_step, new_instruction)
    seen = set()
    uniq = []
    for r in raw:
        key = (r["intersection_id"], r["entry_step"], r["new_instruction"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    # Earlier entries first, then clip to max_k
    uniq.sort(key=lambda d: d["entry_step"])
    uniq = uniq[:max_k]

    onset_manuals = [max(0, r["entry_step"] - ENTRY_LOOKBACK_STEPS) for r in uniq]
    instructions = [f"turn {r['new_instruction']} at the intersection" for r in uniq]
    return instructions, onset_manuals


def build_lane_lines(scenario: SD) -> Dict[str, LineString]:
    """Map lane_id -> centerline from scenario lane polylines."""
    features = scenario[SD.MAP_FEATURES]
    lane_lines: Dict[str, LineString] = {}

    for lane_id, lane_data in features.items():
        if lane_data.get("type") != "LANE_SURFACE_STREET":
            continue
        pts = [tuple(p[:2]) for p in lane_data.get("polyline", [])]
        if len(pts) >= 2:
            try:
                line = LineString(pts)
                if line.length > LANE_MIN_LENGTH:
                    lane_lines[lane_id] = line
            except Exception:
                continue
    return lane_lines


def find_intersections(scenario: SD) -> Optional[List[Polygon]]:
    """Detect intersection-like zones via crosswalks and crossing lane buffers."""
    from math import atan2, degrees

    def heading(line: LineString) -> float:
        x1, y1 = line.coords[0]
        x2, y2 = line.coords[-1]
        return atan2(y2 - y1, x2 - x1)

    def angle_between(l1: LineString, l2: LineString) -> float:
        return abs(degrees(heading(l1) - heading(l2))) % 180

    features = scenario[SD.MAP_FEATURES]
    lane_lines = build_lane_lines(scenario)

    crosswalk_polygons: Dict[str, Polygon] = {}
    for cid, data in features.items():
        if data.get("type") != "CROSSWALK":
            continue
        poly = data.get("polygon", [])
        if len(poly) >= 3:
            try:
                crosswalk_polygons[cid] = Polygon([tuple(p[:2]) for p in poly])
            except Exception:
                continue

    # Start with buffered crosswalks
    intersection_candidates: List[Polygon] = [poly.buffer(3.0) for poly in crosswalk_polygons.values()]

    # Add lane crossing regions (different headings and close)
    lane_ids = list(lane_lines.keys())
    for i in range(len(lane_ids)):
        for j in range(i + 1, len(lane_ids)):
            l1, l2 = lane_lines[lane_ids[i]], lane_lines[lane_ids[j]]
            if l1.distance(l2) < 1.5 and angle_between(l1, l2) > 15:
                region = l1.buffer(1.5).intersection(l2.buffer(1.5))
                if not region.is_empty:
                    intersection_candidates.append(region)

    if not intersection_candidates:
        return None

    fused = unary_union([c.buffer(1.0) for c in intersection_candidates]).buffer(0)
    zones = [fused] if fused.geom_type == "Polygon" else list(fused.geoms)

    # Filter by area and lane/crosswalk coverage
    valid: List[Polygon] = []
    for zone in zones:
        if zone.area < INTERSECTION_MIN_AREA:
            continue
        lanes_intersecting = sum(1 for line in lane_lines.values() if line.intersects(zone))
        if lanes_intersecting >= 3 and any(zone.intersects(cw) for cw in crosswalk_polygons.values()):
            valid.append(zone)

    return sorted(valid, key=lambda z: z.area, reverse=True)[:MAX_INTERSECTION_ZONES] if valid else None


# =========================
# Merge detection (geometry + behavior)
# =========================

def _linestring_heading(line: LineString) -> float:
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[-1]
    return float(np.arctan2(y2 - y1, x2 - x1))


def _angle_deg(a: float, b: float) -> float:
    d = np.degrees(np.arctan2(np.sin(a - b), np.cos(a - b)))
    return abs(float(d))


def _project_point_to_line(pt: np.ndarray, line: LineString) -> Tuple[np.ndarray, float, float]:
    """
    Return (closest_xy, s_along, signed_lateral).
    s_along is cumulative arclength to the projection.
    signed_lateral uses the segment normal to indicate left/right.
    """
    coords = list(line.coords)
    best = None
    best_dist = 1e9
    acc_len = 0.0
    best_s = 0.0

    for i in range(len(coords) - 1):
        p0 = np.array(coords[i], dtype=float)
        p1 = np.array(coords[i + 1], dtype=float)
        v = p1 - p0
        if np.allclose(v, 0):
            continue
        t = float(np.clip(np.dot(pt - p0, v) / np.dot(v, v), 0.0, 1.0))
        proj = p0 + t * v
        d = float(np.linalg.norm(pt - proj))
        if d < best_dist:
            best_dist = d
            best = proj
            best_s = acc_len + float(np.linalg.norm(proj - p0))
        acc_len += float(np.linalg.norm(v))

    signed_lat = 0.0
    if best is not None:
        # Find the nearest segment again to compute a normal
        min_d = 1e9
        best_v = None
        best_seg_proj = None
        for i in range(len(coords) - 1):
            p0 = np.array(coords[i], dtype=float)
            p1 = np.array(coords[i + 1], dtype=float)
            v = p1 - p0
            if np.allclose(v, 0):
                continue
            t = float(np.clip(np.dot(pt - p0, v) / np.dot(v, v), 0.0, 1.0))
            proj = p0 + t * v
            d = float(np.linalg.norm(pt - proj))
            if d < min_d:
                min_d = d
                best_v = v
                best_seg_proj = proj
        if best_v is not None:
            n = np.array([-best_v[1], best_v[0]], dtype=float)
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-6:
                n /= n_norm
                signed_lat = float(np.dot(pt - best_seg_proj, n))

    return (np.array(best, dtype=float) if best is not None else np.array(pt, dtype=float)), float(best_s), float(signed_lat)


def _build_lane_lines_and_tree(scenario: SD) -> Tuple[Dict[str, LineString], List[LineString], Optional[STRtree]]:
    features = scenario[SD.MAP_FEATURES]
    lane_lines: Dict[str, LineString] = {}
    for fid, data in features.items():
        if data.get("type") != "LANE_SURFACE_STREET":
            continue
        pts = [tuple(p[:2]) for p in data.get("polyline", [])]
        if len(pts) >= 2:
            try:
                line = LineString(pts)
                if line.length >= LANE_MIN_LENGTH:
                    lane_lines[fid] = line
            except Exception:
                continue

    lane_list: List[LineString] = list(lane_lines.values())
    tree = STRtree(lane_list) if lane_list else None
    return lane_lines, lane_list, tree



def _candidate_adjacent_lanes(
    ego_line: LineString,
    lane_lines: Dict[str, LineString],
    lane_list: List[LineString],
    tree: Optional[STRtree],
    max_lat: float,
    max_heading_diff_deg: float
) -> List[LineString]:
    if tree is None:
        return []

    box = ego_line.buffer(max_lat + 1.0)
    ego_head = _linestring_heading(ego_line)
    cands: List[LineString] = []

    result = tree.query(box)
    try:
        it = list(result)
    except TypeError:
        it = [result]

    for g in it:
        lane = g if hasattr(g, "geom_type") else lane_list[int(g)]
        if lane.equals(ego_line):
            continue

        head = _linestring_heading(lane)
        if _angle_deg(ego_head, head) <= max_heading_diff_deg:
            pts = [ego_line.interpolate(t, normalized=True) for t in (0.25, 0.5, 0.75)]
            if all(lane.distance(p) <= max_lat for p in pts):
                cands.append(lane)

    return cands



def _merge_zone_from_pair(
    ego_line: LineString,
    target_line: LineString,
    width: float = LANE_BUFFER_WIDTH,
    min_area: float = MERGE_MIN_AREA
) -> Optional[Polygon]:
    poly = ego_line.buffer(width).intersection(target_line.buffer(width))
    if poly.is_empty:
        return None
    polys = [poly] if poly.geom_type == "Polygon" else [g for g in poly.geoms if g.geom_type == "Polygon"]
    polys = [z for z in polys if z.area >= min_area]
    return max(polys, key=lambda z: z.area) if polys else None


def find_merges(scenario: SD) -> Optional[List[Polygon]]:
    """
    Return list of merge zones. Tries a strict pass first; if none found,
    runs a relaxed pass with looser thresholds.
    """
    def run_pass(
        adj_max_lat: float,
        adj_max_head_deg: float,
        min_overlap_area: float,
        lat_vel_thresh: float,
        persist_sec: float,
        delta_d_thresh: float,
        dedup_dist: float,
    ) -> List[Polygon]:
        lane_lines, lane_list, tree = _build_lane_lines_and_tree(scenario)
        if not lane_lines:
            return []

        tracks = scenario.get("tracks", {})
        sdc_id = scenario["metadata"]["sdc_id"]
        ego_pos_seq = np.asarray(tracks[sdc_id]["state"]["position"], dtype=float)
        ts = np.asarray(scenario["metadata"]["ts"], dtype=float)
        dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.1

        # nearest lane + signed lateral offsets
        ego_lane_idx: List[int] = []
        lat_offsets: List[float] = []
        for p in ego_pos_seq:
            point = np.array(p[:2], dtype=float)
            near_idx, near_lane = min(enumerate(lane_list), key=lambda kv: kv[1].distance(Point(point)))
            _, _, d_lat = _project_point_to_line(point, near_lane)
            ego_lane_idx.append(near_idx)
            lat_offsets.append(d_lat)

        lat_offsets = np.asarray(lat_offsets, dtype=float)
        lat_vel = np.zeros_like(lat_offsets)
        lat_vel[1:] = (lat_offsets[1:] - lat_offsets[:-1]) / max(dt, 1e-3)

        # motion tests:
        # (a) instantaneous velocity with persistence
        moving = (np.abs(lat_vel) > lat_vel_thresh).astype(np.int32)
        win = int(max(1, round(persist_sec / max(dt, 1e-3))))
        moving_persist = np.convolve(moving, np.ones(win, dtype=int), mode="same") >= win

        # (b) windowed delta over ~1 s to catch slow steady merges
        w1 = int(max(1, round(1.0 / max(dt, 1e-3))))
        delta_ok = np.zeros_like(lat_offsets, dtype=bool)
        for t in range(len(lat_offsets)):
            t0 = max(0, t - w1)
            if abs(lat_offsets[t] - lat_offsets[t0]) >= delta_d_thresh:
                delta_ok[t] = True

        zones: List[Polygon] = []

        for t in range(1, len(ego_pos_seq)):
            if not (moving_persist[t] or delta_ok[t]):
                continue

            ego_lane = lane_list[ego_lane_idx[t]]
            cands = _candidate_adjacent_lanes(
                ego_line=ego_lane,
                lane_lines=lane_lines,
                lane_list=lane_list,
                tree=tree,
                max_lat=adj_max_lat,
                max_heading_diff_deg=adj_max_head_deg
            )
            if not cands:
                continue

            direction = "left" if lat_offsets[min(t, len(lat_offsets)-1)] - lat_offsets[max(0, t-1)] > 0 else "right"

            point = Point(ego_pos_seq[t][:2])
            _, _, d_e = _project_point_to_line(np.array(point.coords[0], dtype=float), ego_lane)

            # choose nearest candidate on the correct side
            best_lane = None
            best_side_gap = 1e9
            for lane in cands:
                _, _, d_c = _project_point_to_line(np.array(point.coords[0], dtype=float), lane)
                if direction == "left" and d_c <= d_e:
                    continue
                if direction == "right" and d_c >= d_e:
                    continue
                gap = abs(d_c - d_e)
                if gap < best_side_gap:
                    best_side_gap = gap
                    best_lane = lane

            if best_lane is None:
                continue

            zone = _merge_zone_from_pair(ego_lane, best_lane, width=LANE_BUFFER_WIDTH, min_area=min_overlap_area)
            if zone is None:
                continue

            if any(zone.distance(z) < dedup_dist for z in zones):
                continue

            zones.append(zone)
            if len(zones) >= MAX_MERGE_ZONES:
                break

        return zones

    # ---- strict pass (current defaults) ----
    strict_zones = run_pass(
        adj_max_lat=ADJ_MAX_LAT_DIST,           # 5.0
        adj_max_head_deg=ADJ_MAX_HEADING_DIFF_DEG,  # 15
        min_overlap_area=MERGE_MIN_AREA,        # 15 m²
        lat_vel_thresh=LATERAL_VEL_THRESH,      # 0.3 m/s
        persist_sec=LATERAL_PERSIST_SEC,        # 0.5 s
        delta_d_thresh=0.8,                     # need 0.8 m net change over ~1s
        dedup_dist=2.0
    )
    if strict_zones:
        return strict_zones

    # ---- relaxed fallback ----
    relaxed_zones = run_pass(
        adj_max_lat=7.5,            # was 5.0
        adj_max_head_deg=22.0,      # was 15
        min_overlap_area=8.0,       # was 15
        lat_vel_thresh=0.2,         # was 0.3
        persist_sec=0.3,            # was 0.5
        delta_d_thresh=0.6,         # windowed net lateral change
        dedup_dist=3.5              # allow a bit more spread
    )
    return relaxed_zones if relaxed_zones else None




def get_merge_entry_steps(scenario: SD, merge_zones: List[Polygon]) -> List[Tuple[int, int, float]]:
    """Return (merge_id, step_idx, timestamp) when ego enters any merge zone."""
    positions = scenario["tracks"][scenario["metadata"]["sdc_id"]]["state"]["position"]
    timestamps = scenario["metadata"]["ts"]
    entry_info: List[Tuple[int, int, float]] = []
    visited: set = set()

    for i, poly in enumerate(merge_zones):
        for step_idx in range(5, len(positions)):  # skip the very first steps
            pt = Point(positions[step_idx][:2])
            key = (i, step_idx)
            if poly.intersects(pt) and key not in visited:  # intersects is more permissive than contains
                entry_info.append((i, step_idx, float(timestamps[step_idx])))
                visited.add(key)
    return entry_info


def _diversify_directions(cands: List[Dict], max_k: int) -> List[Dict]:
    """
    If both left and right are present, ensure at least one of each appears early.
    Keep original ordering otherwise. Then cap to max_k.
    """
    if not cands:
        return []
    lefts = [c for c in cands if c.get("direction") == "left"]
    rights = [c for c in cands if c.get("direction") == "right"]

    if lefts and rights:
        # Place the earliest left and earliest right at the front (preserving the earlier of the two first)
        first_left = lefts[0]
        first_right = rights[0]
        pair = [first_left, first_right]
        # Remove duplicates from the remaining list
        rest = [c for c in cands if c is not first_left and c is not first_right]
        result = []
        # Keep original temporal order but ensure pair members appear in their original relative order
        # Insert whichever of pair comes earlier first:
        if first_left["entry_step"] <= first_right["entry_step"]:
            result = [first_left, first_right] + rest
        else:
            result = [first_right, first_left] + rest
        return result[:max_k]
    return cands[:max_k]


def select_merge_candidates(
    scenario: SD,
    merge_zones: List[Polygon],
    entry_info: List[Tuple[int, int, float]],
    max_k: int
) -> List[Dict]:
    """Return up to max_k merge choices with adequate lead time, spaced in time and direction-diverse."""
    if not entry_info:
        return []

    positions = scenario["tracks"][scenario["metadata"]["sdc_id"]]["state"]["position"]
    lane_lines = build_lane_lines(scenario)

    # Apply lead requirement
    entry_info = _filter_entries_with_lead(scenario, entry_info,
                                           MIN_TURN_LEAD_STEPS,
                                           MIN_TURN_LEAD_TIME_SEC)
    if not entry_info:
        return []

    # Later entries get a slight preference; tie-breaker: larger merge zone
    entry_info.sort(key=lambda x: (x[1], -merge_zones[x[0]].area))

    # First pass: compute all candidates with directions
    all_cands: List[Dict] = []
    for (best_idx, step, ts) in entry_info:
        ego_pos = np.array(positions[step][:2], dtype=float)
        prev_pos = np.array(positions[max(step - 2, 0)][:2], dtype=float)
        ego_heading = ego_pos - prev_pos
        n = np.linalg.norm(ego_heading)
        ego_heading = ego_heading / n if n > 1e-3 else np.array([1.0, 0.0], dtype=float)

        best_angle = None
        for line in lane_lines.values():
            start = np.array(line.coords[0], dtype=float)
            end = np.array(line.coords[-1], dtype=float)
            if np.linalg.norm(start - ego_pos) < 10.0:
                lane_dir = end - start
                if np.linalg.norm(lane_dir) < 1e-3:
                    continue
                lane_dir = lane_dir / np.linalg.norm(lane_dir)
                angle_deg = float(np.degrees(np.arctan2(np.cross(ego_heading, lane_dir),
                                                        np.dot(ego_heading, lane_dir))))
                if best_angle is None or abs(angle_deg) > abs(best_angle):
                    best_angle = angle_deg

        direction = "left" if (best_angle is not None and best_angle > 0) else "right"

        all_cands.append({
            "type": "merge",
            "merge_id": best_idx,
            "entry_step": step,
            "timestamp": ts,
            "zone_area": merge_zones[best_idx].area,
            "direction": direction
        })

    # Second pass: enforce minimal step spacing to prevent near-duplicates
    spaced: List[Dict] = []
    kept_steps: List[int] = []
    for c in all_cands:
        step = c["entry_step"]
        if any(abs(step - s) < MERGE_MIN_STEP_GAP for s in kept_steps):
            continue
        spaced.append(c)
        kept_steps.append(step)

    # Third: direction diversification (if both sides exist)
    spaced = _diversify_directions(spaced, max_k)

    return spaced[:max_k]


# =========================
# CLI
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="MetaDrive Config Instruction Generation")
    parser.add_argument(
        "-d", "--data_path",
        default=waymo_data,
        type=str,
        help="Path to the scenario dataset root"
    )
    args = parser.parse_args()
    rejects = generate(args.data_path)
    print(f"Scenarios rejected: {rejects}")


if __name__ == "__main__":
    main()
