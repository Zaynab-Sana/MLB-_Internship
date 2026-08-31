"""
evaluate.py
-----------
Loads a saved SimpleCNN checkpoint and evaluates it on the held-out
TEST set (data the model has never seen during training or validation).

Reports:
    - Overall accuracy
    - Precision, Recall, F1-score (macro-averaged, explained below)
    - Per-class precision/recall/F1 for each of the 6 waste categories
    - A confusion matrix image saved to outputs/confusion_matrix/

Run it like this:
    python -m src.evaluate --model models/best_model_adam.pth

--------------------------------------------------------------------
WHY MACRO AVERAGING
--------------------------------------------------------------------
For a multiclass problem, precision/recall/F1 are first computed
separately for each class, then combined into one number. There are
two common ways to combine them:
    - "macro": average the 6 per-class scores with equal weight, so
      every class matters the same regardless of how many images it has.
    - "weighted": average them weighted by how many images each class has,
      so common classes dominate the score.

We report MACRO averaging as the headline number because TrashNet's
classes are not perfectly balanced (e.g. "trash" typically has far
fewer images than "glass" or "paper"). Macro averaging makes sure the
model doesn't get an artificially high score just by doing well on the
biggest classes while ignoring rare ones like "trash" -- which matters
in a real recycling application, since misclassifying rare categories
is still a real-world failure. We also print "weighted" for reference.
--------------------------------------------------------------------
"""

import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import get_device, load_checkpoint, project_root, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SimpleCNN on the TrashNet test set")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=os.path.join("models", "best_model_adam.pth"),
                         help="Path to a saved checkpoint (.pth) from train.py")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Runs the model over an entire DataLoader and collects all predictions and true labels."""
    all_preds, all_labels = [], []
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def evaluate_model(model_path: str, dataset_dir: str = None, batch_size: int = 32,
                    num_workers: int = 2, seed: int = 42):
    set_seed(seed)
    device = get_device()

    # NOTE: we must rebuild the val/test split with the SAME seed used
    # during training, otherwise the "test set" here could accidentally
    # include images the model already saw during training -- this is
    # exactly the kind of data leakage the fixed seed protects against.
    _, _, test_loader, classes = get_dataloaders(
        dataset_dir=dataset_dir, batch_size=batch_size,
        num_workers=num_workers, seed=seed,
    )

    checkpoint_probe = torch.load(model_path, map_location=device)
    num_filters = checkpoint_probe.get("num_filters", 32)
    dropout = checkpoint_probe.get("dropout", 0.5)
    saved_classes = checkpoint_probe.get("classes", classes)

    model = build_model(num_classes=len(saved_classes), num_filters=num_filters, dropout=dropout)
    model, _ = load_checkpoint(model, model_path, device)

    y_true, y_pred = collect_predictions(model, test_loader, device)

    # --- Overall metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\n=== TEST SET RESULTS ===")
    print(f"Accuracy:              {accuracy * 100:.2f}%")
    print(f"Precision (macro):     {precision_macro * 100:.2f}%")
    print(f"Recall (macro):        {recall_macro * 100:.2f}%")
    print(f"F1-score (macro):      {f1_macro * 100:.2f}%")
    print(f"Precision (weighted):  {precision_weighted * 100:.2f}%")
    print(f"Recall (weighted):     {recall_weighted * 100:.2f}%")
    print(f"F1-score (weighted):   {f1_weighted * 100:.2f}%")

    print("\n--- Per-class report ---")
    print(classification_report(y_true, y_pred, target_names=saved_classes, zero_division=0))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=saved_classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title("Confusion Matrix - Test Set")
    plt.tight_layout()

    cm_dir = os.path.join(project_root(), "outputs", "confusion_matrix")
    os.makedirs(cm_dir, exist_ok=True)
    cm_path = os.path.join(cm_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix image saved to: {cm_path}")

    print(
        "\nHow to read the confusion matrix:\n"
        "- Rows = the TRUE class, columns = the PREDICTED class.\n"
        "- The diagonal cells are correct predictions (true positives for that class).\n"
        "- A value in row 'glass', column 'plastic' means: that many actual glass\n"
        "  images were incorrectly predicted as plastic (a false negative for glass,\n"
        "  and a false positive for plastic).\n"
        "- Off-diagonal cells reveal which categories the model commonly confuses --\n"
        "  in TrashNet, glass/plastic and paper/cardboard are common confusion pairs\n"
        "  because they can look visually similar (transparent/white materials, or\n"
        "  similar fibrous textures)."
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "classes": saved_classes,
    }


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        model_path=args.model,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
