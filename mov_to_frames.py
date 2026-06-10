import cv2
import os
import sys

def mov_to_frames(input_path, output_dir="frames", every_n=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: could not open {input_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total}, FPS: {fps:.2f}")

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame_{count:06d}.jpg"), frame)
            saved += 1
        count += 1

    cap.release()
    print(f"Saved {saved} frames to '{output_dir}/'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mov_to_frames.py <input.mov> [output_dir] [every_n_frames]")
        sys.exit(1)

    input_file = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "frames"
    every_n = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    mov_to_frames(input_file, out_dir, every_n)
