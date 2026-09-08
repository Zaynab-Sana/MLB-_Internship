"""
Task 5 — Gradio Demo + ngrok Public URL
=========================================
Requirements:
    pip install ultralytics gradio pyngrok opencv-python

Flow:
    User uploads image -> YOLO model runs inference -> draw boxes/labels/
    confidence on the image -> Gradio displays the result -> ngrok exposes
    this local Gradio server to a public internet URL so it's testable
    from anywhere (e.g. a phone, or a friend's browser), not just localhost.

Run:
    python app.py
    (it will print a public https://xxxx.ngrok-free.app URL in the console)
"""

import cv2
import gradio as gr
from ultralytics import YOLO
from pyngrok import ngrok

WEIGHTS_PATH = "../task3_train_infer/runs/detect/runs_train/custom_yolo/weights/best.pt"
model = YOLO(WEIGHTS_PATH)

# Same fixed palette used in Task 2's visualizer, for consistency across the project
COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0),
]


def detect(image, conf_threshold):
    """
    image: numpy array (RGB) coming from Gradio's image input
    conf_threshold: float from the Gradio slider — lets the user experiment
                     with the precision/recall trade-off live (see Task 3 notes)
    """
    results = model.predict(source=image, conf=conf_threshold, iou=0.45, verbose=False)
    result = results[0]

    annotated = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()

    detections_summary = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        color = COLORS[cls_id % len(COLORS)]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detections_summary.append(f"{label}: {conf:.2f}")

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    summary_text = "\n".join(detections_summary) if detections_summary else "No objects detected."
    return annotated_rgb, summary_text


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="Upload an image"),
        gr.Slider(minimum=0.05, maximum=0.95, value=0.5, step=0.05,
                   label="Confidence threshold"),
    ],
    outputs=[
        gr.Image(type="numpy", label="Detections"),
        gr.Textbox(label="Class : Confidence"),
    ],
    title="Custom YOLO Object Detector",
    description="Upload an image to run the fine-tuned YOLO model. "
                 "Adjust the confidence slider to see the precision/recall trade-off live.",
)


if __name__ == "__main__":
    # --- ngrok setup ---
    from pyngrok import conf
    conf.get_default().ngrok_path = r"C:\Users\TeCh BoSs\AppData\Local\Microsoft\WindowsApps\ngrok.exe"

    PORT = 8660
    public_url = ngrok.connect(PORT)
    print(f"\n Gradio app is live locally on http://127.0.0.1:{PORT}")
    print(f" Public ngrok URL: {public_url}\n")
    print("Share the public URL above to test detection from any device/browser.")

    demo.launch(server_port=PORT, share=False)