"""
BrainHack 2026 — Challenge 1: Perception-only
=============================================
A stripped-down version of challenge1_main.py with ALL flight logic removed
(no MAVSDK, no takeoff, no offboard, no waypoints, no telemetry).

What it does
------------
Runs a single perception pipeline on the RealSense stream:
  1. ArUco detection  → classifies each marker as VALID / INVALID
  2. YOLO detection   → confirms "this is a landing pad" + draws its box
  3. Depth mapping    → feeds each depth frame into GlobalMapper to build a
                        top-down obstacle point cloud

Because there is no drone pose, everything is expressed in the CAMERA-LOCAL
frame (pose fixed at north=0, east=0, yaw=0):
  - "north" = forward distance from the camera (metres)
  - "east"  = lateral offset, positive = right (metres)

On exit (Ctrl+C, or after RUN_SECONDS) it saves:
  - Annotated images of each detected landing pad
  - Top-down obstacle scatter map (PNG)
  - landing_sites_report.txt

Usage
-----
  python challenge1_perception.py
"""

import math
import os
import time
from datetime import datetime

import cv2
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (headless OrangePi)
import matplotlib.pyplot as plt
import numpy as np

from aruco_detector import detect_and_annotate, estimate_marker_ned_offset
from landpad_detecter import (
    load_landpad_model, run_landpad_inference, draw_landpads,
    estimate_landpad_ned_offset,
)
from GlobalMapper import GlobalMapper
from realsense_manager import RealSenseManager

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.expanduser("~/challenge1_output")
RUN_SECONDS = 0          # 0 = run until Ctrl+C; otherwise stop after N seconds
SHOW_WINDOW = False      # True = live preview window (needs a display)

CAM_HEIGHT = 2.0         # camera height above ground, metres (for depth mapping)

# Landing-pad YOLO detector
LANDPAD_MODEL    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_model.pt")
LANDPAD_CONF     = 0.85
LANDPAD_DEVICE   = "cpu"
LANDPAD_EVERY_N  = 1      # run YOLO every Nth frame (raise to spare CPU)
LANDPAD_DEDUPE_M = 0.75   # merge same-class detections within this radius
LANDPAD_SAVE_FALLBACK = 5 # frames to wait for an ArUco verdict before saving anyway

# Fixed pose — no drone, so the map is built in the camera-local frame.
STATIC_POSE = {'north': 0.0, 'east': 0.0, 'yaw': 0.0}


# ─────────────────────────────────────────────────────────────────
# PERCEPTION
# ─────────────────────────────────────────────────────────────────

def perception_frame(rs_mgr, mapper, K, detected_pads, detected_landpads,
                     landpad_model, last_landpad_dets, frame_idx):
    """Process one RealSense frame. Returns (annotated, last_landpad_dets) or
    (None, last_landpad_dets) if no frame was available."""
    color = rs_mgr.get_color()
    depth = rs_mgr.get_depth()
    if color is None or depth is None:
        return None, last_landpad_dets

    # 1) ArUco FIRST — detect + label valid/invalid on a fresh frame.
    annotated, markers = detect_and_annotate(color)

    for det in markers:
        mid = det['id']
        if mid not in detected_pads:
            offset = estimate_marker_ned_offset(det['corners'], depth, K,
                                                cam_height=CAM_HEIGHT)
            if offset is not None:
                pad_n = STATIC_POSE['north'] + offset['north_offset']
                pad_e = STATIC_POSE['east'] + offset['east_offset']
            else:
                pad_n = STATIC_POSE['north']
                pad_e = STATIC_POSE['east']

            detected_pads[mid] = {
                'id':      mid,
                'valid':   det['valid'],
                'north_m': pad_n,
                'east_m':  pad_e,
            }
            print(f"  [ARUCO] ID:{mid} {'VALID' if det['valid'] else 'INVALID'} "
                  f"@ fwd:{pad_n:.2f} right:{pad_e:.2f}")

    # 2) YOLO SECOND — run (throttled) and draw boxes on the same frame.
    if landpad_model is not None and frame_idx % LANDPAD_EVERY_N == 0:
        last_landpad_dets = run_landpad_inference(landpad_model, color, LANDPAD_CONF)
    draw_landpads(annotated, last_landpad_dets)

    # 3) YOLO drives the screenshot. Each landing pad → one save.
    save_this_frame = False
    for d in last_landpad_dets:
        offset = estimate_landpad_ned_offset(d['bbox'], depth, K)
        if offset is None:
            continue
        pad_n = STATIC_POSE['north'] + offset['north_offset']
        pad_e = STATIC_POSE['east'] + offset['east_offset']

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
                  f"@ fwd:{pad_n:.2f} right:{pad_e:.2f}")
        else:
            if d['confidence'] > dup['conf']:
                dup.update(conf=d['confidence'], north_m=pad_n, east_m=pad_e)
            if dup['valid'] is None and verdict is not None:
                dup['valid'], dup['marker_id'] = verdict, verdict_id

        dup['seen'] += 1

        # Save once: as soon as the verdict is known, or after a short fallback.
        if not dup['saved'] and (dup['valid'] is not None
                                 or dup['seen'] >= LANDPAD_SAVE_FALLBACK):
            dup['saved'] = True
            save_this_frame = True

    if save_this_frame:
        ts = datetime.now().strftime("%H%M%S")
        img_path = os.path.join(OUTPUT_DIR, f"landpad_{ts}.jpg")
        cv2.imwrite(img_path, annotated)
        print(f"  [IMG] saved → {img_path}")

    # 4) GlobalMapper update — build the obstacle point cloud.
    mapper.update_frame(depth, STATIC_POSE)

    return annotated, last_landpad_dets


# ─────────────────────────────────────────────────────────────────
# OUTPUT HELPERS
# ─────────────────────────────────────────────────────────────────

def save_map(mapper, detected_pads, out_dir, detected_landpads=None):
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

    ax.set_xlabel("Right [m]  (camera-local east)")
    ax.set_ylabel("Forward [m]  (camera-local north)")
    ax.set_title("Challenge 1 — Camera-Local Obstacle & Landing Pad Map")
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
        f.write("# Positions are camera-local: north=forward, east=right (metres)\n\n")
        f.write("# ArUco markers\n")
        f.write("id,status,forward_m,right_m,distance_m\n")
        for pad in detected_pads.values():
            dist = math.sqrt(pad['north_m']**2 + pad['east_m']**2)
            f.write(f"{pad['id']},"
                    f"{'VALID' if pad['valid'] else 'INVALID'},"
                    f"{pad['north_m']:.3f},"
                    f"{pad['east_m']:.3f},"
                    f"{dist:.3f}\n")

        f.write("\n# Landing pads (YOLO + ArUco verdict)\n")
        f.write("class_name,status,marker_id,confidence,forward_m,right_m,distance_m\n")
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
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── RealSense ──────────────────────────────────────────────
    print("[RS] Starting RealSense...")
    rs_mgr = RealSenseManager()
    warmed = 0
    for _ in range(40):
        try:
            if rs_mgr.grab():
                warmed += 1
        except RuntimeError:
            pass
        if warmed >= 10:
            break
        time.sleep(0.05)
    print(f"[RS] Warmed up ({warmed} frames).")

    K = rs_mgr.K
    mapper = GlobalMapper(K, cam_height=CAM_HEIGHT, obs_h_min=0.1, obs_h_max=1.8,
                          z_min=0.3, z_max=5.0,
                          yaw_in_degrees=True, yaw_clockwise=True,
                          yaw_smoothing=0.7)
    print(f"[RS] K =\n{K}")

    # ── Landing-pad model (YOLO) ───────────────────────────────
    print("[LANDPAD] Loading YOLO model...")
    try:
        landpad_model = load_landpad_model(LANDPAD_MODEL, LANDPAD_DEVICE)
        print(f"[LANDPAD] Model loaded → classes={list(landpad_model.names.values())}")
    except Exception as e:
        print(f"[LANDPAD] Failed to load model ({e}); landing-pad detection OFF.")
        landpad_model = None

    # ── Perception loop ─────────────────────────────────────────
    detected_pads     = {}   # key = marker id
    detected_landpads = []
    last_landpad_dets = []
    frame_idx = 0
    grab_failures = 0

    print("[RUN] Perception running. Press Ctrl+C to stop and save outputs.")
    start_time = time.time()
    try:
        while True:
            if RUN_SECONDS and (time.time() - start_time) >= RUN_SECONDS:
                print(f"[RUN] Reached RUN_SECONDS={RUN_SECONDS}s; stopping.")
                break

            # Grab a frame. wait_for_frames() can raise on a USB/camera hiccup;
            # treat that like an empty grab and keep retrying.
            try:
                ok = rs_mgr.grab()
            except RuntimeError as e:
                ok = False
                if grab_failures == 0:
                    print(f"  [RS] grab error ({e}); retrying...")

            if not ok:
                grab_failures += 1
                if grab_failures % 100 == 0:
                    print(f"  [RS] no frames after {grab_failures} retries; still trying...")
                time.sleep(0.05)
                continue
            grab_failures = 0
            frame_idx += 1

            annotated, last_landpad_dets = perception_frame(
                rs_mgr, mapper, K, detected_pads, detected_landpads,
                landpad_model, last_landpad_dets, frame_idx)

            if SHOW_WINDOW and annotated is not None:
                cv2.imshow("Challenge 1 — Perception", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\n[RUN] Ctrl+C — stopping.")

    elapsed = time.time() - start_time
    print(f"[RUN] Done. {len(detected_pads)} marker(s), "
          f"{len(detected_landpads)} landing pad(s) in {elapsed:.1f}s")

    # ── Save outputs ─────────────────────────────────────────────
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    rs_mgr.stop()
    save_map(mapper, detected_pads, OUTPUT_DIR, detected_landpads)
    save_report(detected_pads, OUTPUT_DIR, detected_landpads)

    print("\n=== Challenge 1 (perception only) complete ===")
    print(f"Detected markers  : {list(detected_pads.keys())}")
    print(f"Detected landpads : {[lp['class_name'] for lp in detected_landpads]}")
    print(f"Output folder     : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()