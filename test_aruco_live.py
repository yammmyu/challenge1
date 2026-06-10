"""
Quick live-webcam test for aruco_detector.detect_and_annotate.

Opens the MacBook webcam, runs ArUco detection on each frame, and shows the
annotated feed with marker boxes + ID/VALID labels in real time.

Run:
    python test_aruco_live.py            # uses camera 0
    python test_aruco_live.py --camera 1 # pick a different camera
    python test_aruco_live.py --list     # list available cameras and exit

Quit: press 'q' or ESC in the video window.

macOS note: if your iPhone shows up instead of the built-in camera, that's
Continuity Camera grabbing a slot. Run with --list to see indices, then pick
the FaceTime/built-in one with --camera N (usually 0, but the iPhone can take
0 when connected).

Markers must come from DICT_7X7_1000 (see aruco_detector.py). Generate one at
https://chev.me/arucogen (set dict = 7x7). Green box = ID in VALID_IDS,
red box = detected but not in the valid set.
"""

import argparse

import cv2

from aruco_detector import detect_and_annotate, VALID_IDS

# AVFoundation is the native macOS capture backend; explicit is more reliable.
_BACKEND = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)


def list_cameras(max_index=5):
    print("Scanning camera indices 0..{}...".format(max_index - 1))
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _BACKEND)
        if cap.isOpened():
            ok, _ = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            status = "ok" if ok else "opened but no frame"
            print(f"  index {i}: available ({w}x{h}, {status})")
            found.append(i)
        cap.release()
    if not found:
        print("  No cameras found.")
    else:
        print(f"Use one with:  python test_aruco_live.py --camera <N>")
    return found


def main(cam_index=0):
    cap = cv2.VideoCapture(cam_index, _BACKEND)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam (index {cam_index}). "
            "Run with --list to see available cameras, and make sure your "
            "terminal/IDE has Camera permission "
            "(System Settings > Privacy & Security > Camera)."
        )

    print(f"Webcam open (index {cam_index}). Valid IDs = {sorted(VALID_IDS)}")
    print("Hold an ArUco marker (DICT_7X7_1000) up to the camera.")
    print("Press 'q' or ESC to quit.")

    # macOS cameras need a moment to warm up; the first reads often fail.
    consecutive_failures = 0
    max_failures = 60  # ~a couple seconds at 30fps before we give up

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(
                    "Giving up: too many failed frame reads. "
                    "Try a different --camera index (run --list)."
                )
                break
            cv2.waitKey(30)  # wait ~30ms and retry instead of quitting
            continue
        consecutive_failures = 0

        annotated, detections = detect_and_annotate(frame)

        # Small HUD showing the current detection count.
        cv2.putText(
            annotated,
            f"markers: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("ArUco live test (press q/ESC to quit)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live ArUco webcam test.")
    parser.add_argument(
        "--camera", type=int, default=0, help="camera index (default 0)"
    )
    parser.add_argument(
        "--list", action="store_true", help="list available cameras and exit"
    )
    args = parser.parse_args()

    if args.list:
        list_cameras()
    else:
        main(args.camera)
