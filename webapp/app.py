"""
webapp/app.py
-------------
A lightweight Flask web app that serves a browser-based UI for the
Smart Waste Classification CNN. Upload or drag in an image, and get a
predicted class + confidence score, with no command line needed.

RUN THIS FROM THE PROJECT ROOT FOLDER (the one containing "src" and
"webapp" as siblings), not from inside webapp/:

    python -m webapp.app

Then open this in your browser:

    http://127.0.0.1:5000
"""

import os
import sys

from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model import build_model
from src.dataset import get_eval_transforms
from src.utils import get_device, load_checkpoint, project_root

app = Flask(__name__)

MODEL_PATH = os.environ.get(
    "WASTE_MODEL_PATH",
    os.path.join(project_root(), "models", "best_model_main_model.pth"),
)

_device = None
_model = None
_classes = None
_transform = get_eval_transforms()


def _load_model_once():
    global _device, _model, _classes
    if _model is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH}'. Train one first with:\n"
            f"    python -m src.train --run_name main_model\n"
            f"or set WASTE_MODEL_PATH to point at your own .pth checkpoint."
        )
    _device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=_device)
    _classes = checkpoint.get("classes")
    if not _classes:
        raise ValueError(
            "This checkpoint has no saved class names -- it wasn't produced "
            "by this project's train.py. Retrain, or point WASTE_MODEL_PATH "
            "at a valid checkpoint."
        )
    num_filters = checkpoint.get("num_filters", 32)
    dropout = checkpoint.get("dropout", 0.5)
    model = build_model(num_classes=len(_classes), num_filters=num_filters, dropout=dropout)
    model, _ = load_checkpoint(model, MODEL_PATH, _device)
    _model = model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        _load_model_once()
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 500

    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image was uploaded."}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "That file couldn't be read as an image. Try a .jpg or .png."}), 400

    input_tensor = _transform(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    ranked = sorted(zip(_classes, probabilities.tolist()), key=lambda pair: -pair[1])
    predicted_class, confidence = ranked[0]

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "all_probabilities": [
            {"class": cls, "probability": round(p * 100, 2)} for cls, p in ranked
        ],
    })


@app.route("/health")
def health():
    try:
        _load_model_once()
        return jsonify({"status": "ok", "model_path": MODEL_PATH, "classes": _classes})
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"status": "model_not_ready", "detail": str(e)}), 503


if __name__ == "__main__":
    print(f"Loading model from: {MODEL_PATH}")
    app.run(host="0.0.0.0", port=5000, debug=False)