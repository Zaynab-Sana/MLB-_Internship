"""
Task 4 — Non-Max Suppression (NMS) from scratch
=================================================
A detector typically outputs MANY overlapping boxes for the same real
object (multiple anchor points / grid cells all "see" the same thing).
NMS cleans that up by keeping only the single best box per object and
discarding the rest.

Algorithm:
  1. Sort all candidate boxes by confidence score, descending.
  2. Take the highest-confidence box, add it to the "keep" list.
  3. Compare it (via IoU) against all remaining boxes of the SAME class.
     Remove any box whose IoU with it exceeds the NMS IoU threshold
     (those are considered "duplicates" of the same object).
  4. Repeat with the next highest-confidence box among what's left.
  5. Stop when no boxes remain.

CONFIDENCE THRESHOLD vs IoU THRESHOLD -- these are two DIFFERENT knobs:
  - Confidence threshold: filters OUT weak detections BEFORE NMS even runs.
    "Is the model sure enough this box contains an object at all?"
  - NMS IoU threshold: decides which overlapping boxes are duplicates of the
    SAME object AFTER we've already decided to trust each box individually.
    "Are these two boxes so overlapping that they must be describing the
    same real-world object?"
  A high NMS IoU threshold (e.g. 0.7) keeps more overlapping boxes (more
  lenient about "these might be two separate close objects"). A low one
  (e.g. 0.3) is aggressive about merging nearby boxes into one.
"""

def compute_iou(box_a, box_b):
    """Same IoU implementation as Task 1 (kept local here so this file runs standalone)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area else 0.0


def nms(boxes, scores, classes, iou_threshold=0.45):
    """
    boxes:   list of (x1, y1, x2, y2) pixel boxes
    scores:  list of confidence scores, same order as boxes
    classes: list of class ids, same order as boxes (NMS is done per-class:
             a 'dog' box should never suppress a 'cat' box even if they overlap)
    iou_threshold: boxes with IoU above this vs. a kept box are removed

    Returns: indices (into the original lists) of boxes to KEEP.
    """
    # Sort indices by score descending
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while order:
        current = order.pop(0)   # highest remaining confidence
        keep.append(current)

        remaining = []
        for idx in order:
            # Only suppress boxes of the SAME class
            if classes[idx] != classes[current]:
                remaining.append(idx)
                continue
            iou = compute_iou(boxes[current], boxes[idx])
            if iou <= iou_threshold:
                remaining.append(idx)   # not a duplicate -> keep considering it
            # else: iou > threshold -> treated as duplicate of `current`, dropped
        order = remaining

    return keep


if __name__ == "__main__":
    # Simulated raw detector output: 4 boxes, 3 of which are duplicates of
    # the same dog, 1 is a separate, non-overlapping cat.
    boxes = [
        (100, 100, 200, 200),  # dog, box A (best)
        (105, 98, 205, 202),   # dog, box B (near-duplicate of A)
        (110, 105, 210, 205),  # dog, box C (near-duplicate of A)
        (400, 400, 500, 500),  # cat, far away, unrelated object
    ]
    scores = [0.91, 0.85, 0.70, 0.88]
    classes = ["dog", "dog", "dog", "cat"]

    for thresh in (0.3, 0.5, 0.7):
        kept = nms(boxes, scores, classes, iou_threshold=thresh)
        print(f"\nNMS IoU threshold = {thresh}")
        print(f"  Boxes kept: {kept} "
              f"({[classes[i] for i in kept]}, scores={[scores[i] for i in kept]})")

"""
Expected pattern when you run this:
  - Low threshold (0.3): aggressive suppression -> only the top-scoring dog
    box (A) and the cat box survive. Any dog box overlapping A by >30% is dropped.
  - High threshold (0.7): more lenient -> some near-duplicate dog boxes may
    survive because their IoU with box A doesn't exceed 0.7.
  - The cat box always survives regardless of threshold: it doesn't overlap
    with any dog box, and NMS never compares across different classes anyway.
"""
