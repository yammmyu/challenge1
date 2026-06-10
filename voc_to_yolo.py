"""
voc_to_yolo.py

Convert the images in ./frames + the Pascal-VOC XML annotations in ./data
into a flat YOLO dataset that the Colab notebook can read and split itself:

    yolo_dataset/
        data.yaml
        images/   <- every frame
        labels/   <- one .txt per frame (empty if not yet labeled)

YOLO label line:  cls cx cy w h   (all normalized 0-1)

Run:  py voc_to_yolo.py
"""

import os
import shutil
import xml.etree.ElementTree as ET

HERE       = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(HERE, "frames")
DATA_DIR   = os.path.join(HERE, "data")
OUTPUT_DIR = os.path.join(HERE, "yolo_dataset")

CLASS_NAMES = ["landing_pad"]                       # add more classes here as they appear
IMG_EXTS    = (".jpg", ".jpeg", ".png", ".bmp")


def parse_voc(xml_path):
    """Pascal-VOC xml -> (width, height, [(name, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name").strip()
        b = obj.find("bndbox")
        boxes.append((name,
                      float(b.findtext("xmin")), float(b.findtext("ymin")),
                      float(b.findtext("xmax")), float(b.findtext("ymax"))))
    return width, height, boxes


def main():
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}

    images_out = os.path.join(OUTPUT_DIR, "images")
    labels_out = os.path.join(OUTPUT_DIR, "labels")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(images_out)
    os.makedirs(labels_out)

    frames = [f for f in os.listdir(FRAMES_DIR)
              if f.lower().endswith(IMG_EXTS)]
    frames.sort()

    n_labeled = 0
    for img in frames:
        stem = os.path.splitext(img)[0]

        # copy the image
        shutil.copy2(os.path.join(FRAMES_DIR, img), os.path.join(images_out, img))

        # build the label (empty if no xml yet)
        lines = []
        xml_path = os.path.join(DATA_DIR, stem + ".xml")
        if os.path.exists(xml_path):
            w, h, boxes = parse_voc(xml_path)
            for name, xmin, ymin, xmax, ymax in boxes:
                if name not in class_to_id:
                    continue
                cx = ((xmin + xmax) / 2.0) / w
                cy = ((ymin + ymax) / 2.0) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                lines.append(f"{class_to_id[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if lines:
                n_labeled += 1

        with open(os.path.join(labels_out, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines))
            if lines:
                fh.write("\n")

    # data.yaml
    names_str = ", ".join(f"'{n}'" for n in CLASS_NAMES)
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as fh:
        fh.write(f"path: {OUTPUT_DIR.replace(os.sep, '/')}\n")
        fh.write("train: images\n")
        fh.write("val: images\n\n")
        fh.write(f"nc: {len(CLASS_NAMES)}\n")
        fh.write(f"names: [{names_str}]\n")

    print(f"Done -> {OUTPUT_DIR}")
    print(f"  {len(frames)} images, {n_labeled} labeled, classes: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
