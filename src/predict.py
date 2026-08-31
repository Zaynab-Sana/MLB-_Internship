"""
predict.py
----------
Loads a trained SimpleCNN and classifies a single new image.

Usage:
    python -m src.predict --image waste.jpg
    python -m src.predict --image waste.jpg --model models/best_model_adam.pth --no_show

Example output:
    Input Image: waste.jpg
    Predicted Class: plastic
    Confidence: 94.70%

--------------------------------------------------------------------
WHY SOFTMAX HERE (AND ONLY HERE)
--------------------------------------------------------------------
During training, CrossEntropyLoss applies Softmax internally, so the
model's forward() function returns raw logits (see model.py). But
logits are not human-readable probabilities -- they can be any real
number, positive or negative. At INFERENCE time, we manually apply
torch.softmax() once, at the very end, purely to turn those logits
into a probability distribution (numbers between 0 and 1 that sum to
1) so we can report something like "94.7% confidence."

IMPORTANT CAVEAT: confidence is the model's own estimate of how sure it
is, not a guarantee of correctness. A model can be confidently wrong,
especially on images that look different from anything in its training
data (different lighting, background clutter, an object it never saw).
Treat confidence as a useful signal, not as ground truth.
--------------------------------------------------------------------
"""

import os
import argparse

import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.model import build_model
from src.dataset import get_eval_transforms
from src.utils import get_device, load_checkpoint, project_root


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the waste category of a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model", type=str, default=os.path.join("models", "best_model_adam.pth"))
    parser.add_argument("--no_show", action="store_true",
                         help="Skip displaying the image (useful on headless servers)")
    parser.add_argument("--save_dir", type=str, default=None,
                         help="Folder to save the annotated prediction image "
                              "(default: outputs/predictions)")
    return parser.parse_args()


def predict_image(image_path: str, model_path: str, show: bool = True, save_dir: str = None):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'. Check the path and try again.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. "
            f"Run train.py first, or point --model at an existing .pth file."
        )

    device = get_device()

    checkpoint_probe = torch.load(model_path, map_location=device)
    classes = checkpoint_probe.get("classes")
    if classes is None:
        raise ValueError(
            "This checkpoint doesn't contain class names. It may have been saved by an "
            "older/incompatible version of train.py."
        )
    num_filters = checkpoint_probe.get("num_filters", 32)
    dropout = checkpoint_probe.get("dropout", 0.5)

    model = build_model(num_classes=len(classes), num_filters=num_filters, dropout=dropout)
    model, _ = load_checkpoint(model, model_path, device)

    # Load and preprocess the image EXACTLY like validation/test images
    # (resize, tensor conversion, normalization -- no random augmentation).
    original_image = Image.open(image_path).convert("RGB")
    transform = get_eval_transforms()
    input_tensor = transform(original_image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        logits = model(input_tensor)                     # raw scores, shape (1, num_classes)
        probabilities = torch.softmax(logits, dim=1)[0]   # convert to probabilities, shape (num_classes,)
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    predicted_class = classes[predicted_idx.item()]
    confidence_percent = confidence.item() * 100

    print(f"Input Image: {os.path.basename(image_path)}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence_percent:.2f}%")

    # Show the full probability breakdown for transparency, not just the top class.
    print("\nAll class probabilities:")
    for cls, prob in sorted(zip(classes, probabilities.tolist()), key=lambda x: -x[1]):
        print(f"  {cls:12s}: {prob * 100:5.2f}%")

    # Save + optionally display the image with the prediction as a title.
    save_dir = save_dir or os.path.join(project_root(), "outputs", "predictions")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"prediction_{os.path.basename(image_path)}.png")

    plt.figure(figsize=(5, 5))
    plt.imshow(original_image)
    plt.title(f"Predicted: {predicted_class} ({confidence_percent:.1f}%)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"\nAnnotated prediction image saved to: {out_path}")

    return predicted_class, confidence_percent


if __name__ == "__main__":
    args = parse_args()
    predict_image(
        image_path=args.image,
        model_path=args.model,
        show=not args.no_show,
        save_dir=args.save_dir,
    )
