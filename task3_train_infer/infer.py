"""
Task 3 — Inference on unseen images + confidence threshold comparison
========================================================================
Confidence threshold = the minimum "how sure am I this box contains an
object of this class" score the model must output before we keep a
detection. It's a knob you set at inference time, not something learned.

Effect of raising it:
  - Fewer, more "confident" boxes are kept
  - False positives DROP (junk/low-confidence guesses get filtered out)
  - False negatives RISE (real objects the model was only somewhat sure about
    get thrown away too)
It's a precision/recall trade-off, not a free lunch.
"""

import os
from ultralytics import YOLO


def run_inference(weights_path, image_dir, out_root="predictions", conf_thresholds=(0.25, 0.5, 0.75)):
    model = YOLO(weights_path)

    for conf in conf_thresholds:
        out_dir = os.path.join(out_root, f"conf_{conf}")
        print(f"\n--- Running inference at confidence threshold = {conf} ---")

        results = model.predict(
            source=image_dir,
            conf=conf,          # <-- the threshold under test
            iou=0.45,           # NMS IoU threshold (kept fixed here; see Task 4 for that knob)
            save=True,
            project=out_root,
            name=f"conf_{conf}",
            exist_ok=True,
        )

        total_boxes = 0
        for r in results:
            total_boxes += len(r.boxes)
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                print(f"  {os.path.basename(r.path)}: class={cls_name} "
                      f"conf={confidence:.2f} box={[round(v, 1) for v in xyxy]}")

        print(f"Total detections kept at conf={conf}: {total_boxes}")
        print(f"Annotated images saved to: {out_dir}/")


if __name__ == "__main__":
    WEIGHTS = "runs/detect/runs_train/custom_yolo/weights/best.pt"
    TEST_IMAGES = "../dataset/images/val"  # or a separate held-out 'unseen' folder

    run_inference(WEIGHTS, TEST_IMAGES, conf_thresholds=(0.25, 0.5, 0.75))

"""
OBSERVATION TEMPLATE (fill in after running on your own dataset):

conf=0.25 -> most detections kept, some are low-quality/wrong (more false positives)
conf=0.50 -> balanced, typically the default sweet spot
conf=0.75 -> only very confident boxes kept, some real objects get missed
             (more false negatives), but remaining boxes are highly reliable

Use case guidance:
  - Safety-critical detection (e.g. don't miss a pedestrian): favor LOWER
    threshold, accept more false positives to avoid missing true objects.
  - High-precision filtering (e.g. auto-tagging where wrong tags are costly):
    favor HIGHER threshold.
"""
