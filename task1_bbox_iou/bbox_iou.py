"""
Task 1 — Bounding Boxes & IoU
=============================
Two common bbox formats:
  1) XYXY  (x_min, y_min, x_max, y_max)      -> pixel coordinates, corner format
  2) XYWH  (x_min, y_min, width, height)     -> pixel coordinates, top-left + size
  3) YOLO  (x_center, y_center, width, height) -> ALL VALUES NORMALIZED (0-1) by image W/H

This script:
  - Converts between all three formats
  - Implements IoU (Intersection over Union) from scratch (no libraries)
  - Tests IoU on several box pairs, including IoU=0 and IoU=1 edge cases
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. FORMAT CONVERSIONS
# ---------------------------------------------------------------------------

def xyxy_to_xywh(box):
    """(x_min, y_min, x_max, y_max) -> (x_min, y_min, w, h)"""
    x_min, y_min, x_max, y_max = box
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def xywh_to_xyxy(box):
    """(x_min, y_min, w, h) -> (x_min, y_min, x_max, y_max)"""
    x_min, y_min, w, h = box
    return (x_min, y_min, x_min + w, y_min + h)


def xyxy_to_yolo(box, img_w, img_h):
    """
    Pixel corner box -> YOLO normalized (x_center, y_center, w, h), all in [0,1].
    This is the format YOLO expects in every .txt annotation file:
        class_id x_center y_center width height
    Why normalize? It makes the label independent of image resolution —
    a box covering the same *relative* area is described the same way
    whether the image is 640x480 or 1920x1080.
    """
    x_min, y_min, x_max, y_max = box
    box_w = x_max - x_min
    box_h = y_max - y_min
    x_center = x_min + box_w / 2
    y_center = y_min + box_h / 2

    return (
        x_center / img_w,
        y_center / img_h,
        box_w / img_w,
        box_h / img_h,
    )


def yolo_to_xyxy(yolo_box, img_w, img_h):
    """YOLO normalized (xc, yc, w, h) -> pixel (x_min, y_min, x_max, y_max)."""
    xc, yc, w, h = yolo_box
    # De-normalize back to pixels first
    xc_px, yc_px = xc * img_w, yc * img_h
    w_px, h_px = w * img_w, h * img_h
    x_min = xc_px - w_px / 2
    y_min = yc_px - h_px / 2
    x_max = xc_px + w_px / 2
    y_max = yc_px + h_px / 2
    return (x_min, y_min, x_max, y_max)


# ---------------------------------------------------------------------------
# 2. IoU FROM SCRATCH
# ---------------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """
    Intersection over Union between two boxes in XYXY pixel format.

    Steps (this is the part people get wrong, so read carefully):
      1. Find the overlapping rectangle between the two boxes by taking the
         MAX of the two x_min/y_min and the MIN of the two x_max/y_max.
      2. If that rectangle is invalid (right edge <= left edge, or
         bottom edge <= top edge), there is NO overlap -> intersection = 0.
      3. Union = area(A) + area(B) - intersection
         (we subtract intersection because it would otherwise be counted twice)
      4. IoU = intersection / union
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Step 1: coordinates of the intersection rectangle
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # Step 2: width/height of intersection; clamp to 0 if boxes don't overlap
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Areas of each box
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    # Step 3: union
    union_area = area_a + area_b - inter_area

    # Step 4: guard against division by zero (both boxes have zero area)
    if union_area == 0:
        return 0.0

    return inter_area / union_area


@dataclass
class TestCase:
    name: str
    box_a: tuple
    box_b: tuple
    expected_note: str


if __name__ == "__main__":
    img_w, img_h = 640, 480  # example image size for normalization demo

    # --- Demonstrate format conversions on one box ---
    sample_xyxy = (100, 50, 300, 250)  # a box drawn/labeled on a 640x480 image
    sample_xywh = xyxy_to_xywh(sample_xyxy)
    sample_yolo = xyxy_to_yolo(sample_xyxy, img_w, img_h)
    back_to_xyxy = yolo_to_xyxy(sample_yolo, img_w, img_h)

    print("=== Format Conversion Demo ===")
    print(f"Original XYXY : {sample_xyxy}")
    print(f"-> XYWH       : {sample_xywh}")
    print(f"-> YOLO norm  : {tuple(round(v, 4) for v in sample_yolo)}")
    print(f"-> back XYXY  : {tuple(round(v, 2) for v in back_to_xyxy)}")

    # --- IoU test cases ---
    tests = [
        TestCase("Identical boxes", (50, 50, 150, 150), (50, 50, 150, 150),
                  "IoU = 1.0 -> boxes are pixel-for-pixel identical, perfect match"),
        TestCase("No overlap at all", (0, 0, 50, 50), (200, 200, 250, 250),
                  "IoU = 0.0 -> boxes don't touch, completely wrong prediction"),
        TestCase("Partial overlap", (50, 50, 150, 150), (100, 100, 200, 200),
                  "IoU between 0 and 1 -> decent but imperfect localization"),
        TestCase("One box fully inside another", (0, 0, 200, 200), (50, 50, 100, 100),
                  "Low IoU even though inner box is fully contained -> union is "
                  "dominated by the big box's area, so IoU stays small"),
        TestCase("Touching edges only", (0, 0, 50, 50), (50, 0, 100, 50),
                  "IoU = 0.0 -> boxes share only a boundary line, zero area overlap"),
    ]

    print("\n=== IoU Test Cases ===")
    for t in tests:
        iou = compute_iou(t.box_a, t.box_b)
        print(f"{t.name:28s} | box_a={t.box_a} box_b={t.box_b} | IoU={iou:.4f}")
        print(f"   -> {t.expected_note}")

"""
EXPLANATION — IoU = 0 vs IoU = 1
---------------------------------
IoU = 1.0 : The predicted box and the ground-truth box are EXACTLY the same
            rectangle (same corners). Intersection area == Union area.
            In practice a real model almost never hits exactly 1.0 — it's the
            theoretical upper bound used as a sanity check for the metric.

IoU = 0.0 : The predicted box and ground-truth box have ZERO overlapping
            area. This happens when boxes don't touch, or only touch along a
            shared edge/corner (a line or point has zero area, so intersection
            area is 0). This is treated as a completely missed detection —
            during evaluation this prediction contributes to false positives
            and the ground truth to false negatives.

In real training, most IoU values will land somewhere between 0 and 1.
Object detectors typically use a threshold (commonly 0.5) — predictions with
IoU >= threshold against a ground-truth box are counted as a "correct" match
(a "True Positive"), otherwise they're a "False Positive".
"""
