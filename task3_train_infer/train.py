"""
Task 3 — Train YOLO on the custom dataset
==========================================
Uses Ultralytics YOLOv8 (pip install ultralytics).
We fine-tune a pretrained checkpoint (transfer learning) rather than training
from scratch — with only 30-50 images, training from random weights would
badly overfit / fail to converge. Starting from COCO-pretrained weights means
the model already knows general edges/shapes/textures and just needs to
adapt its final layers to your specific classes.
"""

from ultralytics import YOLO

def main():
    # 'yolov8n.pt' = nano checkpoint: smallest/fastest, ideal for a small
    # custom dataset and for iterating quickly on a laptop/Colab GPU.
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="../dataset/data.yaml",
        epochs=50,          # small dataset -> more epochs helps, but watch for overfitting
        imgsz=640,          # standard YOLO input resolution
        batch=8,            # lower this if you run out of GPU memory
        patience=15,        # early stopping: stop if val metric doesn't improve for 15 epochs
        project="runs_train",
        name="custom_yolo",
        exist_ok=True,
    )

    print("Training complete.")
    print(f"Best weights saved at: runs_train/custom_yolo/weights/best.pt")

    # Immediately validate on the val split to get Precision/Recall/mAP
    metrics = model.val()
    print(metrics)  # includes box.map, box.map50, box.mp (precision), box.mr (recall)


if __name__ == "__main__":
    main()
