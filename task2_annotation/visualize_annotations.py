"""
Task 2 — Visualize YOLO Annotations
====================================
Draws every bounding box from a YOLO .txt label file onto its matching image,
so you can sanity-check your annotations before training (catch mislabeled
classes, off-by-one boxes, flipped x/y, etc. — very common annotation bugs).

Usage:
    python visualize_annotations.py --images dataset/images/train \
                                     --labels dataset/labels/train \
                                     --classes cat dog bird \
                                     --out visualized/

Requires: opencv-python  (pip install opencv-python)
"""

import os
import argparse
import cv2

# A small fixed color palette so each class is visually distinct and consistent
COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0),
]


def yolo_to_pixels(xc, yc, w, h, img_w, img_h):
    """Undo YOLO normalization -> pixel corner coordinates for drawing."""
    xc_px, yc_px = xc * img_w, yc * img_h
    w_px, h_px = w * img_w, h * img_h
    x1 = int(xc_px - w_px / 2)
    y1 = int(yc_px - h_px / 2)
    x2 = int(xc_px + w_px / 2)
    y2 = int(yc_px + h_px / 2)
    return x1, y1, x2, y2


def visualize_one(image_path, label_path, class_names, out_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [skip] could not read image: {image_path}")
        return
    img_h, img_w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"  [warn] no label file for {image_path}, saving image unchanged")
        cv2.imwrite(out_path, img)
        return

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid, xc, yc, w, h = line.split()
            cid = int(cid)
            xc, yc, w, h = map(float, (xc, yc, w, h))

            x1, y1, x2, y2 = yolo_to_pixels(xc, yc, w, h, img_w, img_h)
            color = COLORS[cid % len(COLORS)]
            label = class_names[cid] if cid < len(class_names) else str(cid)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # Filled label background so text stays readable over any image
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="folder of images")
    parser.add_argument("--labels", required=True, help="folder of matching .txt YOLO labels")
    parser.add_argument("--classes", nargs="+", required=True, help="class names in class_id order")
    parser.add_argument("--out", default="visualized", help="output folder for drawn images")
    args = parser.parse_args()

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    images = [f for f in os.listdir(args.images) if f.lower().endswith(valid_ext)]
    print(f"Found {len(images)} images. Drawing annotations...")

    for fname in images:
        image_path = os.path.join(args.images, fname)
        label_name = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(args.labels, label_name)
        out_path = os.path.join(args.out, fname)
        visualize_one(image_path, label_path, args.classes, out_path)

    print(f"Done. Visualized images saved to: {args.out}/")


if __name__ == "__main__":
    main()
