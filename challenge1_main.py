"""
BrainHack 2026 — Challenge 1: Reconnaissance
=============================================
Runs on the OrangePi compute board of the Mapping Drone.

What it does
------------
1. Takes off to 2 m via offboard mode (MAVSDK, serial connection)
2. Executes a boustrophedon (lawnmower) sweep at ≤ 0.3 m/s
3. At each waypoint: grabs color + depth frame from the RealSense
   - Runs ArUco detection → classifies each landing pad as VALID/INVALID
   - Feeds depth frame into GlobalMapper to build obstacle point cloud
4. Lands, then saves:
   - Annotated images of each detected marker
   - Top-down obstacle scatter map (PNG)
   - landing_sites_report.txt with NED positions + distances

Usage
-----
  python challenge1_main.py

Before running
--------------
  - VALID_IDS in aruco_detector.py must be updated on assessment day
  - WAYPOINTS list below should be tuned to actual cage dimensions
    (set these after your first practice cage run)
"""

import asyncio
import math
import os
import time
from datetime import datetime

import cv2
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (headless OrangePi)
import matplotlib.pyplot as plt
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw

from aruco_detector import detect_and_annotate, estimate_marker_ned_offset
from landpad_detecter import (
    load_landpad_model, run_landpad_inference, draw_landpads,
    estimate_landpad_ned_offset,
)
from GlobalMapper import GlobalMapper
from realsense_manager import RealSenseManager

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION — tune before assessment
# ─────────────────────────────────────────────────────────────────

SERIAL_ADDR   = "serial:///dev/ttyS6:921600"
CRUISE_SPEED  = 0.25    # m/s lateral — must stay ≤ 0.3 m/s (rule)
TAKEOFF_ALT   = -3.5    # NED down (negative = up), metres
HOVER_SECS    = 2.5     # seconds to dwell at each waypoint for detection
WAYPOINT_TOL  = 0.30    # metres: how close counts as "arrived"
OUTPUT_DIR    = os.path.expanduser("~/challenge1_output")

# Landing-pad YOLO detector (combined with ArUco in one perception pass)
LANDPAD_MODEL    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_model.pt")
LANDPAD_CONF     = 0.85
LANDPAD_DEVICE   = "cpu"
LANDPAD_EVERY_N  = 1      # run YOLO every Nth frame (raise to spare CPU)
LANDPAD_DEDUPE_M = 0.75   # merge same-class detections within this radius
LANDPAD_SAVE_FALLBACK = 5 # frames to wait for an ArUco verdict before saving anyway

# Boustrophedon waypoints in LOCAL NED (north_m, east_m, down_m, yaw_deg).
# Tune these to match the cage after your first practice run.
# Pattern: two forward lanes separated by ~2 m, facing North (0°) on odd
# rows and South (180°) on even rows so the camera always faces unscanned area.
WAYPOINTS = [
    # (north_m, east_m, down_m, yaw_deg)
    (0.0,  0.0, TAKEOFF_ALT,  0.0),   # start — hover after takeoff
    (4.0,  0.0, TAKEOFF_ALT,  0.0),   # lane 1 forward
    (4.0,  2.5, TAKEOFF_ALT,  90.0),  # step right
    (0.0,  2.5, TAKEOFF_ALT, 180.0),  # lane 2 backward
    (0.0,  5.0, TAKEOFF_ALT,  90.0),  # step right
    (4.0,  5.0, TAKEOFF_ALT,  0.0),   # lane 3 forward
]

# ─────────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────────

class SharedState:
    def __init__(self):
        self.latest_position = None   # mavsdk PositionNed
        self.latest_yaw      = 0.0    # degrees
        self.stop_flag       = asyncio.Event()


# ─────────────────────────────────────────────────────────────────
# BACKGROUND TELEMETRY MONITOR
# ─────────────────────────────────────────────────────────────────

async def position_monitor(drone: System, state: SharedState):
    async def _pos():
        async for pv in drone.telemetry.position_velocity_ned():
            if state.stop_flag.is_set():
                break
            state.latest_position = pv.position

    async def _yaw():
        async for att in drone.telemetry.attitude_euler():
            if state.stop_flag.is_set():
                break
            state.latest_yaw = att.yaw_deg

    try:
        await asyncio.gather(_pos(), _yaw())
    except asyncio.CancelledError:
        pass


# ─────────────────────────────────────────────────────────────────
# CONTINUOUS PERCEPTION (ArUco + mapping, runs the whole mission)
# ─────────────────────────────────────────────────────────────────

async def perception_loop(rs_mgr, mapper, state, K, detected_pads,
                          detected_landpads, landpad_model=None):
    """One combined perception pipeline, run for the whole flight.

    Per frame:
      1. ArUco FIRST — detect markers on the clean frame, draw VALID/INVALID
         labels, and record each marker. This is the source of truth for a
         landing pad's validity.
      2. YOLO SECOND — confirm "this is a landing pad" and draw its box onto
         the same annotated frame (run in a worker thread so flight control is
         never blocked; the latest boxes stay drawn between runs).
      3. YOLO drives the screenshot — a pad's combined frame is saved once,
         on the frame where its ArUco verdict is known (or after a short
         fallback so pads with no readable marker are still captured).

    Runs continuously so markers and pads are caught while cruising between
    waypoints, not only while hovering at them.
    """
    last_landpad_dets = []   # most recent YOLO boxes, redrawn every frame
    frame_idx = 0
    grab_failures = 0        # consecutive failed/empty grabs (camera hiccup)
    try:
        while not state.stop_flag.is_set():
            # Grab a frame. RealSense wait_for_frames() can RAISE (e.g.
            # "Frame didn't arrive within 5000ms") on a transient USB/camera
            # hiccup. Catch it and treat it like an empty grab so one glitch
            # can't kill perception for the rest of the flight — keep retrying.
            try:
                ok = rs_mgr.grab()
            except RuntimeError as e:
                ok = False
                if grab_failures == 0:
                    print(f"  [RS] grab error ({e}); retrying...")

            color = rs_mgr.get_color() if ok else None
            depth = rs_mgr.get_depth() if ok else None

            if color is None or depth is None:
                grab_failures += 1
                if grab_failures % 100 == 0:
                    print(f"  [RS] no frames after {grab_failures} retries; still trying...")
                await asyncio.sleep(0.05)
                continue
            grab_failures = 0
            frame_idx += 1

            # 1) ArUco FIRST — detect + label valid/invalid on a fresh frame.
            annotated, markers = detect_and_annotate(color)

            for det in markers:
                mid = det['id']
                if mid not in detected_pads:
                    offset = estimate_marker_ned_offset(det['corners'], depth, K, cam_height=2.0)
                    if offset is not None and state.latest_position is not None:
                        pad_n = state.latest_position.north_m + offset['north_offset']
                        pad_e = state.latest_position.east_m  + offset['east_offset']
                    elif state.latest_position is not None:
                        pad_n = state.latest_position.north_m
                        pad_e = state.latest_position.east_m
                    else:
                        continue   # no pose yet — can't localise

                    detected_pads[mid] = {
                        'id':     mid,
                        'valid':  det['valid'],
                        'north_m': pad_n,
                        'east_m':  pad_e,
                    }
                    print(f"  [ARUCO] ID:{mid} {'VALID' if det['valid'] else 'INVALID'} "
                          f"@ N:{pad_n:.2f} E:{pad_e:.2f}")

            # 2) YOLO SECOND — run (throttled) and draw boxes on the same frame.
            if landpad_model is not None and frame_idx % LANDPAD_EVERY_N == 0:
                last_landpad_dets = await asyncio.to_thread(
                    run_landpad_inference, landpad_model, color, LANDPAD_CONF)
            draw_landpads(annotated, last_landpad_dets)

            # 3) YOLO drives the screenshot. Each landing pad → one save.
            save_this_frame = False
            if state.latest_position is not None:
                for d in last_landpad_dets:
                    offset = estimate_landpad_ned_offset(d['bbox'], depth, K)
                    if offset is None:
                        continue
                    pad_n = state.latest_position.north_m + offset['north_offset']
                    pad_e = state.latest_position.east_m  + offset['east_offset']

                    # Validity = ArUco marker whose centre falls inside the box.
                    x1, y1, x2, y2 = d['bbox']
                    verdict, verdict_id = None, None
                    for det in markers:
                        cx, cy = det['center_px']
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            verdict, verdict_id = det['valid'], det['id']
                            break

                    dup = next((p for p in detected_landpads
                                if p['class_name'] == d['class_name']
                                and math.hypot(p['north_m'] - pad_n,
                                               p['east_m'] - pad_e) < LANDPAD_DEDUPE_M),
                               None)
                    if dup is None:
                        dup = {
                            'class_name': d['class_name'],
                            'conf':       d['confidence'],
                            'north_m':    pad_n,
                            'east_m':     pad_e,
                            'valid':      verdict,
                            'marker_id':  verdict_id,
                            'seen':       0,
                            'saved':      False,
                        }
                        detected_landpads.append(dup)
                        print(f"  [LANDPAD] {d['class_name']} conf={d['confidence']:.2f} "
                              f"@ N:{pad_n:.2f} E:{pad_e:.2f}")
                    else:
                        if d['confidence'] > dup['conf']:
                            dup.update(conf=d['confidence'], north_m=pad_n, east_m=pad_e)
                        if dup['valid'] is None and verdict is not None:
                            dup['valid'], dup['marker_id'] = verdict, verdict_id

                    dup['seen'] += 1

                    # Save once: as soon as the verdict is known, or after a
                    # short fallback if no marker ever resolves.
                    if not dup['saved'] and (dup['valid'] is not None
                                             or dup['seen'] >= LANDPAD_SAVE_FALLBACK):
                        dup['saved'] = True
                        save_this_frame = True

            if save_this_frame:
                ts  = datetime.now().strftime("%H%M%S")
                img_path = os.path.join(OUTPUT_DIR, f"landpad_{ts}.jpg")
                cv2.imwrite(img_path, annotated)
                print(f"  [IMG] saved → {img_path}")

            # 5) GlobalMapper update.
            if state.latest_position is not None:
                pose = {
                    'north': state.latest_position.north_m,
                    'east':  state.latest_position.east_m,
                    'yaw':   state.latest_yaw,
                }
                mapper.update_frame(depth, pose)

            await asyncio.sleep(0.02)
    except asyncio.CancelledError:
        pass


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _distance_2d(pos, target_n, target_e):
    return math.sqrt((pos.north_m - target_n)**2 + (pos.east_m - target_e)**2)


async def fly_to(drone: System, state: SharedState,
                 north, east, down, yaw_deg, speed=CRUISE_SPEED):
    """Send a position+velocity setpoint and wait until arrived."""
    # Velocity vector pointing toward target (capped at speed)
    if state.latest_position is not None:
        dn = north - state.latest_position.north_m
        de = east  - state.latest_position.east_m
        mag = math.sqrt(dn**2 + de**2) or 1e-9
        vn = speed * dn / mag
        ve = speed * de / mag
    else:
        vn, ve = 0.0, 0.0

    pos = PositionNedYaw(north_m=north, east_m=east, down_m=down, yaw_deg=yaw_deg)
    vel = VelocityNedYaw(north_m_s=vn, east_m_s=ve, down_m_s=0.0, yaw_deg=yaw_deg)
    await drone.offboard.set_position_velocity_ned(pos, vel)

    # Poll until within tolerance (max 60 s)
    deadline = time.time() + 60.0
    while time.time() < deadline:
        if state.latest_position is not None:
            if _distance_2d(state.latest_position, north, east) < WAYPOINT_TOL:
                break
        await asyncio.sleep(0.2)


# ─────────────────────────────────────────────────────────────────
# OUTPUT HELPERS
# ─────────────────────────────────────────────────────────────────

def save_map(mapper: GlobalMapper, detected_pads, out_dir, detected_landpads=None):
    pts = mapper.get_global_points()
    fig, ax = plt.subplots(figsize=(8, 8))

    if len(pts) > 0:
        dists = np.linalg.norm(pts, axis=1)
        ax.scatter(pts[:, 1], pts[:, 0], c=dists, s=4, cmap='viridis',
                   edgecolors='none', alpha=0.6, label='Obstacles')

    for pad in detected_pads.values():
        marker = '^' if pad['valid'] else 'x'
        clr    = 'green' if pad['valid'] else 'red'
        ax.plot(pad['east_m'], pad['north_m'], marker=marker,
                color=clr, markersize=12,
                label=f"ID:{pad['id']} ({'VALID' if pad['valid'] else 'INVALID'})")

    for lp in (detected_landpads or []):
        valid = lp.get('valid')
        status = 'VALID' if valid else ('INVALID' if valid is False else '?')
        clr = 'green' if valid else ('red' if valid is False else 'gold')
        ax.plot(lp['east_m'], lp['north_m'], marker='*',
                color=clr, markeredgecolor='black', markersize=16,
                label=f"Landpad:{lp['class_name']} {status} ({lp['conf']:.2f})")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Challenge 1 — Top-Down Obstacle & Landing Pad Map")
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    path = os.path.join(out_dir, "top_down_map.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[MAP] saved → {path}")


def save_report(detected_pads, out_dir, detected_landpads=None):
    path = os.path.join(out_dir, "landing_sites_report.txt")
    with open(path, 'w') as f:
        f.write("# ArUco markers\n")
        f.write("id,status,north_m,east_m,distance_from_origin_m\n")
        for pad in detected_pads.values():
            dist = math.sqrt(pad['north_m']**2 + pad['east_m']**2)
            f.write(f"{pad['id']},"
                    f"{'VALID' if pad['valid'] else 'INVALID'},"
                    f"{pad['north_m']:.3f},"
                    f"{pad['east_m']:.3f},"
                    f"{dist:.3f}\n")

        f.write("\n# Landing pads (YOLO + ArUco verdict)\n")
        f.write("class_name,status,marker_id,confidence,north_m,east_m,distance_from_origin_m\n")
        for lp in (detected_landpads or []):
            dist = math.sqrt(lp['north_m']**2 + lp['east_m']**2)
            valid = lp.get('valid')
            status = 'VALID' if valid else ('INVALID' if valid is False else 'UNVERIFIED')
            f.write(f"{lp['class_name']},"
                    f"{status},"
                    f"{lp.get('marker_id', '')},"
                    f"{lp['conf']:.3f},"
                    f"{lp['north_m']:.3f},"
                    f"{lp['east_m']:.3f},"
                    f"{dist:.3f}\n")
    print(f"[REPORT] saved → {path}")


# ─────────────────────────────────────────────────────────────────
# MAIN ASYNC RUN
# ─────────────────────────────────────────────────────────────────

async def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── RealSense ──────────────────────────────────────────────
    print("[RS] Starting RealSense...")
    rs_mgr = RealSenseManager()
    # Warm up — discard first few frames. Tolerate a slow camera start: the
    # first grab() calls can raise before the stream is producing frames.
    warmed = 0
    for _ in range(40):
        try:
            if rs_mgr.grab():
                warmed += 1
        except RuntimeError:
            pass
        if warmed >= 10:
            break
        await asyncio.sleep(0.05)
    print(f"[RS] Warmed up ({warmed} frames).")

    K      = rs_mgr.K
    mapper = GlobalMapper(K, cam_height=2.0, obs_h_min=0.1, obs_h_max=1.8,
                          z_min=0.3, z_max=5.0,
                          yaw_in_degrees=True, yaw_clockwise=True,
                          yaw_smoothing=0.7)
    print(f"[RS] K =\n{K}")

    # ── Landing-pad model (YOLO) ───────────────────────────────
    # Loaded once; inference runs inside perception_loop (in a worker thread)
    # so it shares one annotated frame with ArUco. Disabled gracefully if the
    # weights are missing.
    print("[LANDPAD] Loading YOLO model...")
    try:
        landpad_model = load_landpad_model(LANDPAD_MODEL, LANDPAD_DEVICE)
        print(f"[LANDPAD] Model loaded → classes={list(landpad_model.names.values())}")
    except Exception as e:
        print(f"[LANDPAD] Failed to load model ({e}); landing-pad detection OFF.")
        landpad_model = None

    # ── MAVSDK ─────────────────────────────────────────────────
    print("[DRONE] Connecting...")
    drone = System()
    await drone.connect(system_address=SERIAL_ADDR)

    async for conn in drone.core.connection_state():
        if conn.is_connected:
            print("[DRONE] Connected.")
            break

    print("[DRONE] Waiting for optical flow / local position lock...")
    async for health in drone.telemetry.health():
        if health.is_local_position_ok:
            print("[DRONE] Local position healthy.")
            break

    # ── Shared state + position monitor ────────────────────────
    state   = SharedState()
    mon_task = asyncio.create_task(position_monitor(drone, state))

    # Wait for first telemetry
    for _ in range(50):
        if state.latest_position is not None:
            break
        await asyncio.sleep(0.1)

    # ── Arm + takeoff ───────────────────────────────────────────
    print(f"[DRONE] Setting takeoff altitude {abs(TAKEOFF_ALT)} m...")
    await drone.action.set_takeoff_altitude(abs(TAKEOFF_ALT))

    print("[DRONE] Arming...")
    await drone.action.arm()

    # Pre-stream mandatory setpoint before offboard
    init_pos = PositionNedYaw(0.0, 0.0, 0.0, 0.0)
    init_vel = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
    await drone.offboard.set_position_velocity_ned(init_pos, init_vel)

    print("[DRONE] Starting offboard mode...")
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"[ERROR] Offboard start failed: {e}")
        await drone.action.disarm()
        return

    print("[DRONE] Taking off to 2 m...")
    takeoff_pos = PositionNedYaw(0.0, 0.0, TAKEOFF_ALT, 0.0)
    takeoff_vel = VelocityNedYaw(0.0, 0.0, -0.5, 0.0)   # -0.5 m/s ascent
    await drone.offboard.set_position_velocity_ned(takeoff_pos, takeoff_vel)
    await asyncio.sleep(6)  # let it stabilise at altitude

    # ── Continuous perception (runs for the whole sweep) ────────
    detected_pads     = {}   # key = marker id, value = {id, valid, north_m, east_m}
    detected_landpads = []   # [{class_name, conf, valid, marker_id, north_m, east_m, ...}]
    perception_task = asyncio.create_task(
        perception_loop(rs_mgr, mapper, state, K, detected_pads,
                        detected_landpads, landpad_model))

    # ── Sweep loop ──────────────────────────────────────────────
    print("[SWEEP] Starting boustrophedon sweep...")
    start_time = time.time()

    for wp_idx, (wn, we, wd, wyaw) in enumerate(WAYPOINTS):
        print(f"[WP {wp_idx+1}/{len(WAYPOINTS)}] → N:{wn:.1f} E:{we:.1f} Yaw:{wyaw:.0f}°")
        await fly_to(drone, state, wn, we, wd, wyaw)

        # Brief dwell at the waypoint for a clean look; detection + mapping
        # keep running continuously in perception_loop (also while cruising).
        await asyncio.sleep(HOVER_SECS)

    elapsed = time.time() - start_time
    print(f"[SWEEP] Done. {len(detected_pads)}/5 pads, "
          f"{len(detected_landpads)} landing pad(s) found in {elapsed:.1f}s")

    # ── Stop perception before landing ──────────────────────────
    perception_task.cancel()
    try:
        await perception_task
    except asyncio.CancelledError:
        pass

    # ── Land ────────────────────────────────────────────────────
    print("[DRONE] Stopping offboard, landing...")
    try:
        await drone.offboard.stop()
    except OffboardError:
        pass
    await drone.action.land()
    await asyncio.sleep(8)

    # ── Stop telemetry monitor ──────────────────────────────────
    state.stop_flag.set()
    mon_task.cancel()
    try:
        await mon_task
    except asyncio.CancelledError:
        pass

    # ── Save outputs ─────────────────────────────────────────────
    rs_mgr.stop()
    save_map(mapper, detected_pads, OUTPUT_DIR, detected_landpads)
    save_report(detected_pads, OUTPUT_DIR, detected_landpads)

    print("\n=== Challenge 1 complete ===")
    print(f"Detected pads     : {list(detected_pads.keys())}")
    print(f"Detected landpads : {[lp['class_name'] for lp in detected_landpads]}")
    print(f"Output folder     : {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(run())
