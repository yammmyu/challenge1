import pyrealsense2 as rs
import numpy as np
import threading


class RealSenseManager:
    """
    Thread-safe wrapper around pyrealsense2 for simultaneous color + aligned depth streaming.
    Replaces the Gazebo-only depth_receiver.py for use on the real mapping drone.
    """

    def __init__(self, color_w=640, color_h=480, fps=30):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, color_w, color_h, rs.format.z16, fps)

        self.align = rs.align(rs.stream.color)
        self.profile = self.pipeline.start(cfg)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        self._K = self._build_K()
        self._lock = threading.Lock()
        self._color = None
        self._depth = None  # float32, metres

    def _build_K(self):
        intr = (self.profile
                    .get_stream(rs.stream.color)
                    .as_video_stream_profile()
                    .get_intrinsics())
        return np.array([[intr.fx, 0.0,     intr.ppx],
                         [0.0,     intr.fy,  intr.ppy],
                         [0.0,     0.0,      1.0     ]], dtype=np.float64)

    # ------------------------------------------------------------------
    # Call this once per control-loop tick (blocking, ~33 ms at 30 fps)
    # ------------------------------------------------------------------
    def grab(self):
        frameset = self.pipeline.wait_for_frames()
        aligned  = self.align.process(frameset)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return False

        color = np.asanyarray(color_frame.get_data())
        raw   = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth = raw * self._depth_scale  # convert raw uint16 → metres

        with self._lock:
            self._color = color
            self._depth = depth
        return True

    # ------------------------------------------------------------------
    # Thread-safe accessors
    # ------------------------------------------------------------------
    @property
    def K(self):
        return self._K

    def get_color(self):
        with self._lock:
            return None if self._color is None else self._color.copy()

    def get_depth(self):
        with self._lock:
            return None if self._depth is None else self._depth.copy()

    def stop(self):
        self.pipeline.stop()
