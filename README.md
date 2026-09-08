# Module 15 — Day 2: Object Detection with YOLO

## Project structure
```
yolo_module15/
├── task1_bbox_iou/
│   └── bbox_iou.py              # bbox format conversions + IoU from scratch
├── task2_annotation/
│   ├── annotate_utils.py        # write/read YOLO .txt annotations, multi-object handling
│   └── visualize_annotations.py # draws saved annotations back onto images
├── task3_train_infer/
│   ├── train.py                 # fine-tunes YOLOv8n on the custom dataset
│   └── infer.py                 # inference at multiple confidence thresholds
├── task4_nms_eval/
│   ├── nms.py                   # Non-Max Suppression from scratch
│   └── evaluate.py              # Precision / Recall / mAP@50 / mAP@50:95
├── task5_gradio/
│   └── app.py                   # Gradio UI + ngrok public URL
├── dataset/
│   ├── images/{train,val}/      # <- put your 30-50 collected images here
│   ├── labels/{train,val}/      # <- matching YOLO .txt files (from annotation tool)
│   └── data.yaml                # class names + dataset paths
└── README.md
```

## Setup
```bash
pip install ultralytics opencv-python gradio pyngrok
```

## Approach

1. **Collect & annotate images** (Task 2). I chose 2–3 classes and gathered
   30–50 images per the assignment. Annotation itself was done with a GUI
   tool (LabelImg), which exports directly in YOLO format — one `.txt` per
   image, one line per object. `annotate_utils.py` documents that format and
   also lets you script-generate labels if you already have pixel boxes.
   `visualize_annotations.py` draws the saved boxes back onto the images so
   annotation mistakes (flipped coordinates, wrong class id) are caught
   before training wastes time on bad data.

2. **Bounding boxes & IoU** (Task 1). Implemented conversions between
   XYXY, XYWH, and YOLO's normalized format, and IoU purely with basic
   arithmetic (no library) to make sure the underlying geometry is
   understood, not just called via `torchvision.ops.box_iou`.

3. **Training** (Task 3). Fine-tuned a COCO-pretrained `yolov8n.pt` on the
   custom dataset (transfer learning) — training from scratch would badly
   overfit on only ~40 images. Ran inference on a held-out validation split
   at three confidence thresholds (0.25 / 0.5 / 0.75) to see the effect
   directly.

4. **NMS & Evaluation** (Task 4). Implemented NMS from scratch to show how
   duplicate/overlapping boxes for the same object get collapsed into one,
   then used Ultralytics' validator (which implements the standard COCO-style
   protocol) to report Precision, Recall, mAP@50, and mAP@50:95.

5. **Gradio + ngrok demo** (Task 5). Wrapped the trained model in a Gradio
   interface with an image upload and a live confidence-threshold slider,
   then exposed it publicly with ngrok so it's testable outside localhost.

## The complete flow, end to end

```
Image
  │  (camera / dataset)
  ▼
Annotation
  │  human/tool draws boxes + assigns class -> saved as YOLO .txt
  ▼
Bounding Box
  │  stored as normalized (x_center, y_center, w, h); converted to pixel
  │  XYXY when needed for drawing/IoU math
  ▼
YOLO Prediction
  │  trained model outputs many candidate boxes per image, each with a
  │  class id + box coordinates + confidence score
  ▼
Confidence (threshold)
  │  discard predictions below the chosen confidence cutoff
  │  (knob: raises precision, can lower recall)
  ▼
IoU
  │  used in TWO places:
  │    (a) NMS — comparing predicted boxes against EACH OTHER
  │    (b) Evaluation — comparing predicted boxes against GROUND TRUTH
  ▼
NMS (Non-Max Suppression)
  │  removes duplicate overlapping boxes for the same object, keeping only
  │  the highest-confidence one per cluster (per class)
  ▼
Final Detection
  │  the clean, de-duplicated set of boxes+labels+scores shown to the user
  ▼
Precision / Recall
  │  computed by matching final detections to ground-truth boxes using IoU
  │  >= 0.5 (or other threshold) to decide True Positive vs False Positive/Negative
  ▼
mAP (mAP@50, mAP@50:95)
  │  summarizes precision/recall across all classes and (for 50:95) across
  │  a sweep of IoU strictness levels, into one comparable number
  ▼
Gradio
  │  wraps the trained model + all of the above (conf threshold, NMS, drawing)
  │  into an interactive web UI: upload image -> see boxes/labels/scores
  ▼
ngrok
     tunnels the local Gradio server to a public HTTPS URL so anyone,
     anywhere, can open it in a browser and test the model — no deployment
     infrastructure required.
```

## Results & observations (fill in with your actual run numbers)

| Confidence threshold | Detections kept | Qualitative effect |
|---|---|---|
| 0.25 | most | more false positives, fewer missed objects |
| 0.50 | balanced | default sweet spot |
| 0.75 | fewest | high-confidence only, some real objects missed |

| Metric | Value |
|---|---|
| Precision | _fill in from evaluate.py_ |
| Recall | _fill in_ |
| mAP@50 | _fill in_ |
| mAP@50:95 | _fill in_ |

**Why accuracy alone isn't enough:** detection has no fixed number of
"predictions to grade" (images contain a variable number of objects),
involves two independent failure modes (wrong class vs. wrong box position),
and a naive accuracy number hides class imbalance. Precision, Recall, and
mAP were built specifically to address these gaps — see the detailed
explanation in `task4_nms_eval/evaluate.py`.

## Public demo
- Gradio app: `task5_gradio/app.py`
- ngrok public URL: _fill in after running `python app.py` — printed to console_
