"""
BrainHack 2026 — Challenge 1: Perception module
===============================================
Reusable perception pipeline, decoupled from flight control.

This is the integration surface for the rest of the team. It owns the camera
and the models; the caller owns the drone and supplies the pose. One call per
control-loop tick does: grab → ArUco detect → YOLO detect → screenshot →
depth-map update → distance estimates.

Camera ownership
----------------
The FLIGHT side owns the camera (it shares the same RealSense pipe as the video
stream). It grabs frames and hands them in; this module never opens the camera.
The only thing it needs from the camera at construction is the intrinsics `K`.

Quick start
-----------
    from realsense_manager import RealSenseManager
    from perception_module import PerceptionModule

    cam  = RealSenseManager()                 # flight side owns this
    perc = PerceptionModule(cam.K, cam_height=2.0)   # loads YOLO only

    # ... each tick, with the drone's current NED pose:
    cam.grab()
    result = perc.process_frame(cam.get_color(), cam.get_depth(),
                                pose={'north': n, 'east': e, 'yaw': yaw_deg})
    for m in result['markers']:
        print(m['id'], m['valid'], m['distance_m'])

    # ... at the end:
    perc.save_outputs()      # writes map PNG + report

Pose
----
`pose` is a dict {'north': float_m, 'east': float_m, 'yaw': float_deg}.
Pass None (or omit) to work in the CAMERA-LOCAL frame (north=forward, east=right).

Threading
---------
`process_frame()` is blocking (YOLO inference, tens of ms). If the caller runs an
asyncio flight loop, offload it so the event loop is never blocked:

    result = await asyncio.to_thread(perc.process_frame, color, depth, pose)
"""

import math
import os
from datetime import datetime

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aruco_detector import detect_and_annotate, estimate_marker_ned_offset
from landpad_detecter import (
    load_landpad_model, run_landpad_inference, draw_landpads,
    estimate_landpad_ned_offset,
)
from GlobalMapper import GlobalMapper


class PerceptionModule:
    """ArUco + YOLO + depth-mapping, behind one ``process_frame()`` call.

    Does NOT own the camera — the flight side grabs frames and passes them in.
    Pass the camera intrinsics ``K`` (e.g. ``RealSenseManager().K``) at construction.
    """

    def __init__(self,
                 K,
                 output_dir="~/challenge1_output",
                 cam_height=2.0,
                 model_path=None,
                 device="cpu",
                 conf=0.85,
                 yolo_every_n=1,
                 dedupe_m=0.75,
                 save_fallback=5,
                 enable_yolo=True,
                 save_screenshots=True):
        self.output_dir = os.path.expanduser(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.K = K
        self.cam_height = cam_height
        self.conf = conf
        self.yolo_every_n = max(1, yolo_every_n)
        self.dedupe_m = dedupe_m
        self.save_fallback = save_fallback
        self.save_screenshots = save_screenshots

        # ── Depth mapper ───────────────────────────────────────
        self.mapper = GlobalMapper(
            self.K, cam_height=cam_height, obs_h_min=0.1, obs_h_max=1.8,
            z_min=0.3, z_max=5.0,
            yaw_in_degrees=True, yaw_clockwise=True, yaw_smoothing=0.7)

        # ── YOLO landing-pad model ─────────────────────────────
        self.model = None
        if enable_yolo:
            if model_path is None:
                model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "my_model.pt")
            try:
                self.model = load_landpad_model(model_path, device)
            except Exception as e:
                print(f"[PERC] YOLO disabled — failed to load model: {e}")

        # ── Accumulated state (persists across ticks) ──────────
        self.markers = {}          # id -> {id, valid, north_m, east_m, distance_m}
        self.landpads = []         # [{class_name, conf, valid, marker_id, north_m, east_m, ...}]
        self._last_yolo = []       # latest YOLO boxes, redrawn between throttled runs
        self._frame_idx = 0

    # ──────────────────────────────────────────────────────────
    # Per-tick entry point
    # ──────────────────────────────────────────────────────────
    def process_frame(self, color, depth, pose=None):
        """Run the full pipeline on a color + aligned-depth pair.

        The caller (flight side) grabs these from the camera each tick. If
        ``color`` or ``depth`` is None (a camera hiccup), returns None — just
        call again next tick.

        Returns
        -------
        dict with:
          'annotated'    : BGR image with ArUco + YOLO boxes drawn
          'depth'        : the depth frame used (float32 metres)
          'markers'      : list of markers THIS frame
                           [{id, valid, center_px, forward_m, right_m, distance_m}]
          'landpads'     : list of landing pads THIS frame
                           [{class_name, confidence, valid, marker_id,
                             forward_m, right_m, distance_m}]
          'screenshot'   : path of a saved screenshot, or None
        """
        if color is None or depth is None:
            return None

        n, e, yaw = self._unpack_pose(pose)
        self._frame_idx += 1

        # 1) ArUco FIRST — detect + label on a clean frame.
        annotated, raw_markers = detect_and_annotate(color)
        frame_markers = []
        for det in raw_markers:
            off = estimate_marker_ned_offset(det['corners'], depth, self.K,
                                             cam_height=self.cam_height)
            fwd = off['north_offset'] if off else 0.0
            rgt = off['east_offset'] if off else 0.0
            dist = off['distance'] if off else None
            frame_markers.append({
                'id':         det['id'],
                'valid':      det['valid'],
                'center_px':  det['center_px'],
                'forward_m':  fwd,
                'right_m':    rgt,
                'distance_m': dist,
            })
            mid = det['id']
            if mid not in self.markers:
                self.markers[mid] = {
                    'id':         mid,
                    'valid':      det['valid'],
                    'north_m':    n + fwd,
                    'east_m':     e + rgt,
                    'distance_m': dist,
                }

        # 2) YOLO SECOND — throttled, drawn on the same frame.
        if self.model is not None and self._frame_idx % self.yolo_every_n == 0:
            self._last_yolo = run_landpad_inference(self.model, color, self.conf)
        draw_landpads(annotated, self._last_yolo)

        # 3) Landing pads → distances, validity, dedup, screenshot.
        frame_landpads = []
        save_path = None
        for d in self._last_yolo:
            off = estimate_landpad_ned_offset(d['bbox'], depth, self.K)
            if off is None:
                continue
            fwd, rgt, dist = off['north_offset'], off['east_offset'], off['distance']
            pad_n, pad_e = n + fwd, e + rgt

            # Validity from any ArUco marker centred inside the box.
            x1, y1, x2, y2 = d['bbox']
            verdict, verdict_id = None, None
            for det in raw_markers:
                cx, cy = det['center_px']
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    verdict, verdict_id = det['valid'], det['id']
                    break

            frame_landpads.append({
                'class_name': d['class_name'],
                'confidence': d['confidence'],
                'valid':      verdict,
                'marker_id':  verdict_id,
                'forward_m':  fwd,
                'right_m':    rgt,
                'distance_m': dist,
            })

            dup = next((p for p in self.landpads
                        if p['class_name'] == d['class_name']
                        and math.hypot(p['north_m'] - pad_n,
                                       p['east_m'] - pad_e) < self.dedupe_m), None)
            if dup is None:
                dup = {
                    'class_name': d['class_name'], 'conf': d['confidence'],
                    'north_m': pad_n, 'east_m': pad_e, 'distance_m': dist,
                    'valid': verdict, 'marker_id': verdict_id,
                    'seen': 0, 'saved': False,
                }
                self.landpads.append(dup)
            else:
                if d['confidence'] > dup['conf']:
                    dup.update(conf=d['confidence'], north_m=pad_n,
                               east_m=pad_e, distance_m=dist)
                if dup['valid'] is None and verdict is not None:
                    dup['valid'], dup['marker_id'] = verdict, verdict_id
            dup['seen'] += 1

            if (self.save_screenshots and not dup['saved']
                    and (dup['valid'] is not None or dup['seen'] >= self.save_fallback)):
                dup['saved'] = True
                ts = datetime.now().strftime("%H%M%S")
                save_path = os.path.join(self.output_dir, f"landpad_{ts}.jpg")
                cv2.imwrite(save_path, annotated)

        # 4) Depth map update.
        self.mapper.update_frame(depth, {'north': n, 'east': e, 'yaw': yaw})

        return {
            'annotated':  annotated,
            'depth':      depth,
            'markers':    frame_markers,
            'landpads':   frame_landpads,
            'screenshot': save_path,
        }

    # ──────────────────────────────────────────────────────────
    # Read-only accessors
    # ──────────────────────────────────────────────────────────
    def get_point_cloud(self):
        """Accumulated obstacle points, (N, 2) array of [north, east] metres."""
        return self.mapper.get_global_points()

    def get_markers(self):
        """All unique ArUco markers seen so far (list of dicts)."""
        return list(self.markers.values())

    def get_landpads(self):
        """All unique landing pads seen so far (list of dicts)."""
        return list(self.landpads)

    # ──────────────────────────────────────────────────────────
    # Outputs
    # ──────────────────────────────────────────────────────────
    def save_outputs(self):
        """Write the top-down map PNG and the landing-sites report."""
        self._save_map()
        self._save_report()

    @staticmethod
    def _unpack_pose(pose):
        if pose is None:
            return 0.0, 0.0, 0.0
        return (float(pose.get('north', 0.0)),
                float(pose.get('east', 0.0)),
                float(pose.get('yaw', 0.0)))

    def _save_map(self):
        pts = self.mapper.get_global_points()
        fig, ax = plt.subplots(figsize=(8, 8))
        if len(pts) > 0:
            dists = np.linalg.norm(pts, axis=1)
            ax.scatter(pts[:, 1], pts[:, 0], c=dists, s=4, cmap='viridis',
                       edgecolors='none', alpha=0.6, label='Obstacles')
        for pad in self.markers.values():
            marker = '^' if pad['valid'] else 'x'
            clr = 'green' if pad['valid'] else 'red'
            ax.plot(pad['east_m'], pad['north_m'], marker=marker, color=clr,
                    markersize=12,
                    label=f"ID:{pad['id']} ({'VALID' if pad['valid'] else 'INVALID'})")
        for lp in self.landpads:
            valid = lp.get('valid')
            status = 'VALID' if valid else ('INVALID' if valid is False else '?')
            clr = 'green' if valid else ('red' if valid is False else 'gold')
            ax.plot(lp['east_m'], lp['north_m'], marker='*', color=clr,
                    markeredgecolor='black', markersize=16,
                    label=f"Landpad:{lp['class_name']} {status} ({lp['conf']:.2f})")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_title("Challenge 1 — Top-Down Obstacle & Landing Pad Map")
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=7)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "top_down_map.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[PERC] map saved → {path}")

    def _save_report(self):
        path = os.path.join(self.output_dir, "landing_sites_report.txt")
        with open(path, 'w') as f:
            f.write("# ArUco markers\n")
            f.write("id,status,north_m,east_m,distance_m\n")
            for pad in self.markers.values():
                d = pad.get('distance_m')
                d = math.sqrt(pad['north_m']**2 + pad['east_m']**2) if d is None else d
                f.write(f"{pad['id']},{'VALID' if pad['valid'] else 'INVALID'},"
                        f"{pad['north_m']:.3f},{pad['east_m']:.3f},{d:.3f}\n")
            f.write("\n# Landing pads (YOLO + ArUco verdict)\n")
            f.write("class_name,status,marker_id,confidence,north_m,east_m,distance_m\n")
            for lp in self.landpads:
                valid = lp.get('valid')
                status = 'VALID' if valid else ('INVALID' if valid is False else 'UNVERIFIED')
                d = lp.get('distance_m')
                d = math.sqrt(lp['north_m']**2 + lp['east_m']**2) if d is None else d
                f.write(f"{lp['class_name']},{status},{lp.get('marker_id', '')},"
                        f"{lp['conf']:.3f},{lp['north_m']:.3f},{lp['east_m']:.3f},{d:.3f}\n")
        print(f"[PERC] report saved → {path}")
