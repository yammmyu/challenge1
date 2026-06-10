#!/usr/bin/env python3
"""Point the camera at your marker; this reports which ArUco dictionary detects it.

Useful when live_aruco.py finds nothing — your marker is probably from a
different dictionary than DICT_7X7_1000. Run this, hold the marker up, and read
the console for the dictionary name + IDs found.

    .venv/bin/python which_dict.py
Press 'q' or ESC to quit.
"""
import cv2

DICTS = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}

detectors = {
    name: cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(val), cv2.aruco.DetectorParameters()
    )
    for name, val in DICTS.items()
}


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera index 0")

    cv2.namedWindow("which_dict", cv2.WINDOW_NORMAL)
    print(f"Scanning {len(detectors)} dictionaries. Hold up your marker...")
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hits = []
        for name, det in detectors.items():
            corners, ids, _ = det.detectMarkers(gray)
            if ids is not None:
                hits.append((name, sorted(int(i) for i in ids.flatten())))

        msg = "  ".join(f"{n}:{i}" for n, i in hits) if hits else "(no markers)"
        if msg != last:
            print(msg)
            last = msg

        cv2.putText(frame, hits[0][0] if hits else "no markers", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("which_dict", frame)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
