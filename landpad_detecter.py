import os
import cv2
import time
import queue
import threading
import numpy as np
from typing import Callable, Optional, Any, List, Dict

from ultralytics import YOLO


# ─────────────────────────────────────────────
#  Detector (unchanged logic from Detector.py)
# ─────────────────────────────────────────────

class Detector:
    def __init__(
        self,
        model_path: str = "my_model.pt",
        confidence_threshold: float = 0.85,
        callback: Optional[Callable[[List[Dict[str, Any]], np.ndarray, Optional[Any]], None]] = None,
        num_workers: int = 1,
        device: str = "cpu",
        save_dir: str = "./detected_images",
        enable_display: bool = True,
        display_window_name: str = "YOLO Detections"
    ):
        self.model = YOLO(model_path).to(device)
        self.conf_threshold = confidence_threshold
        self.callback = callback
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.workers = []

        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self._file_counter = 0
        self._counter_lock = threading.Lock()

        self.enable_display = enable_display
        self.display_window_name = display_window_name
        # maxsize=1 → always show the latest frame, drop stale ones
        self.display_queue = queue.Queue(maxsize=1)
        self.display_thread = None

        self._start_workers(num_workers)

        if self.enable_display:
            self.display_thread = threading.Thread(target=self._display_worker, daemon=True)
            self.display_thread.start()

    def _start_workers(self, num_workers: int) -> None:
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=False)
            t.start()
            self.workers.append(t)

    def submit_image(self, image: np.ndarray, context: Optional[Dict[str, Any]] = None) -> None:
        if context is None:
            context = {}
        self.queue.put((image, context))

    def _get_next_filename(self) -> str:
        with self._counter_lock:
            self._file_counter += 1
            ts = int(time.time() * 1000)
            return f"det_{self._file_counter}_{ts}.jpg"

    def _display_worker(self) -> None:
        cv2.namedWindow(self.display_window_name, cv2.WINDOW_AUTOSIZE)
        while not self.stop_event.is_set():
            try:
                img = self.display_queue.get(timeout=0.05)
                if img is not None:
                    cv2.imshow(self.display_window_name, img)
                    cv2.waitKey(1)
            except queue.Empty:
                continue
            except cv2.error as e:
                print(f"[Display] CV2 error: {e}")
                break
        cv2.destroyWindow(self.display_window_name)

    def _worker(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                image, context = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                results = self.model(image, verbose=False, conf=self.conf_threshold)

                detections = []
                annotated_image = None
                has_detections = False

                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        has_detections = True
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                            conf = float(box.conf[0].cpu().item())
                            cls_id = int(box.cls[0].cpu().item())
                            detections.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": conf,
                                "class_id": cls_id,
                                "class_name": self.model.names[cls_id]
                            })
                        annotated_image = result.plot()

                if has_detections and annotated_image is not None:
                    filename = self._get_next_filename()
                    filepath = os.path.join(self.save_dir, filename)
                    cv2.imwrite(filepath, annotated_image)
                    context["saved_path"] = filepath

                    if self.enable_display:
                        try:
                            self.display_queue.put_nowait(annotated_image)
                        except queue.Full:
                            pass

                    if self.callback:
                        try:
                            self.callback(detections, annotated_image, context)
                        except Exception as e:
                            print(f"[Detector] Callback error: {e}")

            except Exception as e:
                print(f"[Detector] Inference error: {e}")
            finally:
                self.queue.task_done()
                del image, context, results, detections, annotated_image

    def stop(self) -> None:
        self.stop_event.set()
        for t in self.workers:
            t.join()
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join()
        print("[Detector] All workers and display thread stopped.")

    def set_display(self, enabled: bool) -> None:
        self.enable_display = enabled


# ─────────────────────────────────────────────
#  Lightweight functional API (combined pipeline)
# ─────────────────────────────────────────────
# Used by challenge1_main.perception_loop, which runs YOLO + ArUco on the same
# frame and saves a single combined screenshot. The threaded Detector class
# above is kept for standalone / offline use.

def load_landpad_model(model_path: str = "my_model.pt", device: str = "cpu") -> YOLO:
    """Load the YOLO landing-pad model once, ready for repeated inference."""
    return YOLO(model_path).to(device)


def run_landpad_inference(model: YOLO, image: np.ndarray,
                          conf_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """Run YOLO on one BGR frame.

    Returns a list of dicts:
        [{'bbox':[x1,y1,x2,y2], 'confidence':float,
          'class_id':int, 'class_name':str}, ...]
    """
    results = model(image, verbose=False, conf=conf_threshold)
    detections: List[Dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0].cpu().item())
            detections.append({
                'bbox':       [x1, y1, x2, y2],
                'confidence': float(box.conf[0].cpu().item()),
                'class_id':   cls_id,
                'class_name': model.names[cls_id],
            })
    return detections


def draw_landpads(canvas: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw landing-pad bounding boxes + labels onto canvas, in place."""
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d['bbox']]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.putText(canvas, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return canvas


def estimate_landpad_ned_offset(bbox: List[float], depth_img: np.ndarray, K: np.ndarray
                                ) -> Optional[Dict[str, float]]:
    """Bounding-box centre + aligned depth → forward/right offset in body frame.

    Mirrors aruco_detector.estimate_marker_ned_offset but for a YOLO bbox.

    Returns {'north_offset', 'east_offset', 'distance'} or None if the depth
    sample at the bbox centre is missing/out of range.
    """
    x1, y1, x2, y2 = bbox
    cx_px = int((x1 + x2) / 2)
    cy_px = int((y1 + y2) / 2)

    h, w = depth_img.shape
    if not (0 <= cy_px < h and 0 <= cx_px < w):
        return None

    z = float(depth_img[cy_px, cx_px])   # forward distance, metres
    if z <= 0.01 or z > 15.0:
        return None

    fx  = K[0, 0]
    ppx = K[0, 2]
    x_cam = (cx_px - ppx) * z / fx       # lateral (east in body frame, +right)

    return {
        'north_offset': float(z),
        'east_offset':  float(x_cam),
        'distance':     float(np.sqrt(x_cam ** 2 + z ** 2)),
    }


def detection_callback(detections: List[Dict[str, Any]], annotated_image: np.ndarray, context: Any) -> None:
    """Default callback — logs detections. Override via Detector(callback=...)."""
    saved = context.get('saved_path', 'N/A') if isinstance(context, dict) else 'N/A'
    print(f"[Landpad] {len(detections)} detection(s) — saved to {saved}")
    for d in detections:
        print(f"  · {d['class_name']:20s} conf={d['confidence']:.2f} "
              f"bbox={[round(v) for v in d['bbox']]}")
