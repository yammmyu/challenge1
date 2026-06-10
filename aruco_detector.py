import cv2
import numpy as np

# IDs provided by organiser for the 5 landing pads.
# UPDATE this set on assessment day once valid/invalid split is announced.
# Currently: all 5 known marker IDs are treated as potentially valid.
VALID_IDS = {11, 45, 51, 67, 101, 449}

_aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
_detector   = cv2.aruco.ArucoDetector(_aruco_dict, cv2.aruco.DetectorParameters())


def detect_and_annotate(color_img, canvas=None):
    """
    Run ArUco detection on a BGR color image.

    Detection always uses the clean ``color_img``. Marker boxes + labels are
    drawn onto ``canvas`` when provided (e.g. a frame already annotated with
    landing-pad boxes), otherwise onto a fresh copy of ``color_img``. This lets
    the ArUco and landing-pad detectors share one combined annotated frame.

    Returns
    -------
    annotated : ndarray  — the canvas with boxes + labels drawn
    detections : list of dicts  — [{id, valid, corners, center_px}]
    """
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector.detectMarkers(gray)

    annotated   = color_img.copy() if canvas is None else canvas
    detections  = []

    if ids is None:
        return annotated, detections

    for corner, marker_id in zip(corners, ids.flatten()):
        is_valid = int(marker_id) in VALID_IDS
        box_color = (0, 255, 0) if is_valid else (0, 0, 255)
        label     = f"ID:{marker_id} {'VALID' if is_valid else 'INVALID'}"

        pts = corner[0].astype(int)
        cv2.polylines(annotated, [pts], isClosed=True, color=box_color, thickness=2)

        center = pts.mean(axis=0).astype(int)

        # Background rectangle: full marker width, 1/3 of marker height, centered vertically
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        cy = int(center[1])
        band_half = (y_max - y_min) // 6  # 1/3 height total, so 1/6 each side of center
        bg_tl = (int(x_min), cy - band_half)
        bg_br = (int(x_max), cy + band_half)
        cv2.rectangle(annotated, bg_tl, bg_br, box_color, thickness=cv2.FILLED)

        # Centered text inside the background band
        font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, 2
        text_color = (0, 0, 0)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_org = (int(center[0]) - tw // 2, cy + th // 2)
        cv2.putText(annotated, label, text_org,
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        detections.append({
            'id':        int(marker_id),
            'valid':     is_valid,
            'corners':   corner,
            'center_px': (int(center[0]), int(center[1])),
        })

    return annotated, detections


def estimate_marker_ned_offset(corners, depth_img, K, cam_height=1.0):
    """
    Given a single marker's corners and the aligned depth image,
    return the (north_offset, east_offset, distance) of the marker
    relative to the drone in the forward-right camera frame.

    north_offset: forward distance (Z_cam in camera optical frame)
    east_offset:  lateral distance (X_cam, positive = right)
    distance:     Euclidean distance from camera
    """
    pts = corners[0].astype(np.float32)
    cx_px, cy_px = pts.mean(axis=0)
    cx_px, cy_px = int(cx_px), int(cy_px)

    h, w = depth_img.shape
    if not (0 <= cy_px < h and 0 <= cx_px < w):
        return None

    z = float(depth_img[cy_px, cx_px])
    if z <= 0.01 or z > 15.0:
        return None

    fx, fy = K[0, 0], K[1, 1]
    ppx, ppy = K[0, 2], K[1, 2]

    x_cam = (cx_px - ppx) * z / fx   # lateral (east in body frame)
    z_fwd = z                          # forward (north in body frame)
    dist  = np.sqrt(x_cam**2 + z_fwd**2)

    return {
        'north_offset': float(z_fwd),
        'east_offset':  float(x_cam),
        'distance':     float(dist),
    }
