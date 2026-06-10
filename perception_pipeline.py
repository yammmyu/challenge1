"""
perception_pipeline.py
======================
Self-contained perception module for Challenge 1.

Runs ArUco detection + YOLO landing-pad inference + GlobalMapper updates
continuously from a RealSense camera stream.

Integration
-----------
Typical usage from a teammate's main loop:

    from perception_pipeline import PerceptionPipeline

    pipeline = PerceptionPipeline(
        rs_mgr       = rs_mgr,        # RealSenseManager instance
        mapper       = mapper,         # GlobalMapper instance
        landpad_model= landpad_model,  # YOLO model or None
        output_dir   = "~/output",     # where annotated JPEGs are saved
    )

    # Start in the background (pass your SharedState-compatible object)
    task = asyncio.create_task(pipeline.run(state))

    # Read results at any time (thread-safe):
    frame       = pipeline.latest_frame      # np.ndarray BGR, or None
    pads        = pipeline.detected_pads     # dict  {marker_id: {...}}
    landpads    = pipeline.detected_landpads # list  [{class_name, ...}]

    # Stop cleanly:
    task.cancel()
    await task
"""

import asyncio
import math
import os
import threading
from datetime import datetime

import cv2
import numpy as np

from aruco_detector import detect_and_annotate, estimate_marker_ned_offset
from landpad_detecter import (
    run_landpad_inference, draw_landpads, estimate_landpad_ned_offset,
)

# ── Defaults (override via PerceptionPipeline.__init__ kwargs) ────────────────

DEFAULT_LANDPAD_CONF       = 0.85
DEFAULT_LANDPAD_EVERY_N    = 1      # run YOLO every Nth frame
DEFAULT_LANDPAD_DEDUPE_M   = 0.75   # merge detections within this radius (m)
DEFAULT_LANDPAD_SAVE_FALLBACK = 5   # frames before saving a pad with no ArUco verdict


class PerceptionPipeline:
    """
    Wraps the ArUco + YOLO + mapping perception loop.

    Parameters
    ----------
    rs_mgr          RealSenseManager — provides .grab(), .get_color(),
                    .get_depth(), and .K
    mapper          GlobalMapper — receives mapper.update_frame(depth, pose)
    landpad_model   YOLO model returned by load_landpad_model(), or None to
                    disable landing-pad detection
    output_dir      Directory where annotated JPEG snapshots are saved
    landpad_conf    YOLO confidence threshold
    landpad_every_n Run YOLO inference every Nth frame (raise to spare CPU)
    landpad_dedupe_m Radius in metres below which same-class pads are merged
    landpad_save_fallback
                    Frames to wait for an ArUco verdict before saving anyway
    """

    def __init__(
        self,
        rs_mgr,
        mapper,
        landpad_model=None,
        output_dir="~/perception_output",
        landpad_conf=DEFAULT_LANDPAD_CONF,
        landpad_every_n=DEFAULT_LANDPAD_EVERY_N,
        landpad_dedupe_m=DEFAULT_LANDPAD_DEDUPE_M,
        landpad_save_fallback=DEFAULT_LANDPAD_SAVE_FALLBACK,
    ):
        self._rs_mgr         = rs_mgr
        self._mapper         = mapper
        self._landpad_model  = landpad_model
        self._output_dir     = os.path.expanduser(output_dir)
        self._conf           = landpad_conf
        self._every_n        = landpad_every_n
        self._dedupe_m       = landpad_dedupe_m
        self._save_fallback  = landpad_save_fallback

        # ── Public result containers (read from outside at any time) ──────────
        self.detected_pads     = {}   # {marker_id: {id, valid, north_m, east_m}}
        self.detected_landpads = []   # [{class_name, conf, valid, marker_id, ...}]

        # ── Latest annotated frame (thread-safe) ──────────────────────────────
        self._frame_lock  = threading.Lock()
        self._latest_frame = None

    # ── Public frame accessor ─────────────────────────────────────────────────

    @property
    def latest_frame(self):
        """Most recent annotated BGR frame as np.ndarray, or None."""
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    # ── Main coroutine ────────────────────────────────────────────────────────

    async def run(self, state):
        """
        Run the perception loop until cancelled.

        Parameters
        ----------
        state   Object with attributes:
                  .latest_position  — object with .north_m / .east_m, or None
                  .latest_yaw       — float, degrees
                  .stop_flag        — asyncio.Event (optional; loop also stops
                                      on task cancellation)
        """
        os.makedirs(self._output_dir, exist_ok=True)

        K = self._rs_mgr.K
        last_landpad_dets = []
        frame_idx         = 0
        grab_failures     = 0

        try:
            while True:
                # Respect an external stop flag if the caller uses one
                if hasattr(state, 'stop_flag') and state.stop_flag.is_set():
                    break

                # ── Grab frame ────────────────────────────────────────────────
                try:
                    ok = self._rs_mgr.grab()
                except RuntimeError as e:
                    ok = False
                    if grab_failures == 0:
                        print(f"  [RS] grab error ({e}); retrying...")

                color = self._rs_mgr.get_color() if ok else None
                depth = self._rs_mgr.get_depth() if ok else None

                if color is None or depth is None:
                    grab_failures += 1
                    if grab_failures % 100 == 0:
                        print(f"  [RS] no frames after {grab_failures} retries; still trying...")
                    await asyncio.sleep(0.05)
                    continue

                grab_failures = 0
                frame_idx += 1

                # ── 1) ArUco detection ────────────────────────────────────────
                annotated, markers = detect_and_annotate(color)

                for det in markers:
                    mid = det['id']
                    if mid not in self.detected_pads:
                        offset = estimate_marker_ned_offset(
                            det['corners'], depth, K, cam_height=2.0)
                        pos = getattr(state, 'latest_position', None)
                        if offset is not None and pos is not None:
                            pad_n = pos.north_m + offset['north_offset']
                            pad_e = pos.east_m  + offset['east_offset']
                        elif pos is not None:
                            pad_n = pos.north_m
                            pad_e = pos.east_m
                        else:
                            continue

                        self.detected_pads[mid] = {
                            'id':      mid,
                            'valid':   det['valid'],
                            'north_m': pad_n,
                            'east_m':  pad_e,
                        }
                        print(f"  [ARUCO] ID:{mid} "
                              f"{'VALID' if det['valid'] else 'INVALID'} "
                              f"@ N:{pad_n:.2f} E:{pad_e:.2f}")

                # ── 2) YOLO landing-pad detection ─────────────────────────────
                if self._landpad_model is not None and frame_idx % self._every_n == 0:
                    last_landpad_dets = await asyncio.to_thread(
                        run_landpad_inference, self._landpad_model, color, self._conf)
                draw_landpads(annotated, last_landpad_dets)

                # ── 3) Save annotated snapshot when a new pad is confirmed ─────
                save_this_frame = False
                pos = getattr(state, 'latest_position', None)
                if pos is not None:
                    for d in last_landpad_dets:
                        offset = estimate_landpad_ned_offset(d['bbox'], depth, K)
                        if offset is None:
                            continue
                        pad_n = pos.north_m + offset['north_offset']
                        pad_e = pos.east_m  + offset['east_offset']

                        x1, y1, x2, y2 = d['bbox']
                        verdict, verdict_id = None, None
                        for det in markers:
                            cx, cy = det['center_px']
                            if x1 <= cx <= x2 and y1 <= cy <= y2:
                                verdict, verdict_id = det['valid'], det['id']
                                break

                        dup = next(
                            (p for p in self.detected_landpads
                             if p['class_name'] == d['class_name']
                             and math.hypot(p['north_m'] - pad_n,
                                            p['east_m']  - pad_e) < self._dedupe_m),
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
                            self.detected_landpads.append(dup)
                            print(f"  [LANDPAD] {d['class_name']} "
                                  f"conf={d['confidence']:.2f} "
                                  f"@ N:{pad_n:.2f} E:{pad_e:.2f}")
                        else:
                            if d['confidence'] > dup['conf']:
                                dup.update(conf=d['confidence'],
                                           north_m=pad_n, east_m=pad_e)
                            if dup['valid'] is None and verdict is not None:
                                dup['valid'], dup['marker_id'] = verdict, verdict_id

                        dup['seen'] += 1

                        if not dup['saved'] and (
                                dup['valid'] is not None
                                or dup['seen'] >= self._save_fallback):
                            dup['saved']    = True
                            save_this_frame = True

                if save_this_frame:
                    ts       = datetime.now().strftime("%H%M%S")
                    img_path = os.path.join(self._output_dir, f"landpad_{ts}.jpg")
                    cv2.imwrite(img_path, annotated)
                    print(f"  [IMG] saved → {img_path}")

                # ── 4) Publish latest annotated frame ─────────────────────────
                with self._frame_lock:
                    self._latest_frame = annotated

                # ── 5) GlobalMapper update ────────────────────────────────────
                if pos is not None:
                    self._mapper.update_frame(depth, {
                        'north': pos.north_m,
                        'east':  pos.east_m,
                        'yaw':   getattr(state, 'latest_yaw', 0.0),
                    })

                await asyncio.sleep(0.02)

        except asyncio.CancelledError:
            pass
