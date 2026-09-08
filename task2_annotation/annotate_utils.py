"""
Task 2 — YOLO Annotation (helper utilities)
============================================
Real annotation of 30-50 images is normally done with a GUI tool, not by
hand-typing coordinates. Recommended free tools:
    - LabelImg      (https://github.com/heartexlabs/labelImg)   -> simplest, saves YOLO .txt directly
    - CVAT          (https://www.cvat.ai/)                       -> browser based, good for teams
    - Roboflow      (https://roboflow.com/)                      -> also handles augmentation/export

Whichever tool you use, you'll end up with the standard YOLO folder layout:

    dataset/
      images/
        train/  img001.jpg, img002.jpg, ...
        val/    img031.jpg, ...
      labels/
        train/  img001.txt, img002.txt, ...   <- one .txt per image, same filename
        val/    img031.txt, ...
      data.yaml

Each line in a label .txt file is ONE object:
    class_id x_center y_center width height
All four numeric values are normalized to [0, 1] (see Task 1's xyxy_to_yolo).
Multiple objects in the same image = multiple lines in that image's .txt file.

This script provides:
  1) write_yolo_annotation()  -> programmatically write a label file (useful if
     you already have pixel boxes, e.g. from an auto-labeler or Task 1 output)
  2) read_yolo_annotation()   -> parse a label file back into a list of boxes
"""

import os


def write_yolo_annotation(label_path, objects, img_w, img_h):
    """
    objects: list of dicts, each like:
        {"class_id": 0, "box_xyxy": (x_min, y_min, x_max, y_max)}
    Converts each box to YOLO format and writes one line per object.
    Multiple objects -> multiple lines -> this is how multi-object images
    are handled: there's no special syntax, just one row per object.
    """
    from math import isfinite

    lines = []
    for obj in objects:
        cid = obj["class_id"]
        x_min, y_min, x_max, y_max = obj["box_xyxy"]

        box_w = x_max - x_min
        box_h = y_max - y_min
        xc = (x_min + box_w / 2) / img_w
        yc = (y_min + box_h / 2) / img_h
        w_norm = box_w / img_w
        h_norm = box_h / img_h

        assert all(isfinite(v) and 0.0 <= v <= 1.0 for v in (xc, yc, w_norm, h_norm)), \
            f"Box out of image bounds after normalization: {obj}"

        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def read_yolo_annotation(label_path):
    """Returns list of (class_id, xc, yc, w, h) tuples, all floats/int normalized."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cid = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            boxes.append((cid, xc, yc, w, h))
    return boxes


if __name__ == "__main__":
    # Example: a 640x480 image with 2 objects (multi-object handling demo)
    demo_objects = [
        {"class_id": 0, "box_xyxy": (50, 60, 200, 220)},   # e.g. class 0 = "cat"
        {"class_id": 1, "box_xyxy": (300, 100, 500, 400)}, # e.g. class 1 = "dog"
    ]
    write_yolo_annotation("demo_labels/img001.txt", demo_objects, img_w=640, img_h=480)
    print("Wrote demo_labels/img001.txt:")
    print(open("demo_labels/img001.txt").read())

    parsed = read_yolo_annotation("demo_labels/img001.txt")
    print("Parsed back:", parsed)

    # class names should live in data.yaml, e.g.:
    # names: ['cat', 'dog', 'bird']
    # so class_id 0 -> 'cat', 1 -> 'dog', 2 -> 'bird'
