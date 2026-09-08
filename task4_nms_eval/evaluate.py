"""
Task 4 — Evaluation: Precision, Recall, mAP@50, mAP@50:95
============================================================
We use Ultralytics' built-in validator (it implements the standard COCO-style
evaluation), rather than reimplementing mAP from scratch, since matching the
exact official algorithm (all confidence/IoU threshold sweeps, per-class then
averaged) is what real benchmarks use.

Definitions:
  Precision = TP / (TP + FP)
      "Of all the boxes I predicted, how many were actually correct?"
      Low precision = model cries wolf a lot (many false alarms).

  Recall = TP / (TP + FN)
      "Of all the real objects that existed, how many did I actually find?"
      Low recall = model misses a lot of real objects.

  mAP@50 = mean Average Precision at a single IoU threshold of 0.5
      A prediction counts as a True Positive only if IoU with its matched
      ground truth is >= 0.5. "Average Precision" (AP) is the area under
      the precision-recall curve for one class; "mean" AP averages that
      across all classes.

  mAP@50:95 = mean AP averaged over IoU thresholds 0.50, 0.55, 0.60, ..., 0.95
      (10 thresholds). This is a stricter, more holistic metric: to score
      well here, a model needs to be well-localized (tight boxes), not
      just roughly-in-the-right-place.

WHY ACCURACY ALONE ISN'T ENOUGH FOR OBJECT DETECTION:
  "Accuracy" (as used in simple classification: correct / total) doesn't
  make sense here for a few reasons:
    1. There's no fixed number of "predictions to grade" — an image can
       contain 0, 1, or 20 objects, and the model can output any number of
       boxes. There's no single denominator to compute % correct against.
    2. Detection has TWO separate things that can go wrong independently:
       classification (right label?) AND localization (right position/size?).
       A model could label everything correctly but draw sloppy boxes —
       accuracy alone wouldn't reveal that, but mAP@50:95 would (it drops
       once box tightness is required).
    3. Class imbalance: if 95% of objects in your dataset are one class, a
       model that mostly guesses that class could look "accurate" while
       being useless on the other classes. Precision/Recall per class (and
       their average, mAP) exposes that; a single accuracy number hides it.
  That's why detection is reported with Precision, Recall, and mAP instead.
"""

from ultralytics import YOLO


def evaluate(weights_path, data_yaml):
    model = YOLO(weights_path)

    # split="val" uses the val set defined in data.yaml
    metrics = model.val(data=data_yaml, split="val")

    print("\n=== Evaluation Results ===")
    print(f"Precision (mean, box.mp) : {metrics.box.mp:.4f}")
    print(f"Recall    (mean, box.mr) : {metrics.box.mr:.4f}")
    print(f"mAP@50                   : {metrics.box.map50:.4f}")
    print(f"mAP@50:95                : {metrics.box.map:.4f}")

    # Per-class breakdown, useful for spotting weak classes
    print("\nPer-class AP@50:")
    for i, name in model.names.items():
        try:
            print(f"  {name:15s}: {metrics.box.ap50[i]:.4f}")
        except (IndexError, KeyError):
            pass

    return metrics


if __name__ == "__main__":
    evaluate(
        weights_path="../task3_train_infer/runs/detect/runs_train/custom_yolo/weights/best.pt",
        data_yaml="../dataset/data.yaml",
    )
