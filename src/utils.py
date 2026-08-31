"""
utils.py
--------
Small shared helper functions used across the project:
- Setting a random seed (reproducibility)
- Picking GPU or CPU automatically
- Saving / loading model checkpoints
- Plotting and saving training curves

Keeping these in one file avoids repeating the same code in
train.py, evaluate.py and predict.py.
"""

import os
import random
import json

import numpy as np
import torch
import matplotlib.pyplot as plt


# The six waste categories, in a fixed order.
# This exact order MUST match the order used everywhere else
# (dataset folder alphabetical order, model output layer, predict.py).
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def set_seed(seed: int = 42) -> None:
    """
    Fix every source of randomness we can, so that re-running the
    code produces the same train/val/test split, the same weight
    initialization, and (as much as PyTorch allows) the same results.

    Why this matters: without a fixed seed, every run would shuffle
    the dataset differently and initialize the network differently,
    making it impossible to fairly compare experiments (e.g. SGD vs Adam).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes cuDNN deterministic at a small performance cost. Fine for
    # a small educational project.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Automatically use a GPU if one is available, otherwise fall back to CPU.
    This means the exact same code works on a laptop with no GPU and on a
    machine with CUDA installed.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU found. Using CPU (training will be slower).")
    return device


def save_checkpoint(model, path: str, extra_info: dict = None) -> None:
    """
    Save a trained model's weights to disk.

    We save only `state_dict()` (the learned numbers), not the whole
    Python object. This is the recommended PyTorch practice because it
    is smaller, more portable, and does not break if the class code
    changes slightly later.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {"model_state_dict": model.state_dict()}
    if extra_info:
        checkpoint.update(extra_info)
    torch.save(checkpoint, path)
    print(f"Model checkpoint saved to: {path}")


def load_checkpoint(model, path: str, device: torch.device):
    """
    Load previously saved weights into a model instance.
    `model` must already be created with the SAME architecture used
    during training, otherwise the shapes will not match.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint found at '{path}'. "
            f"Did you run train.py first to create a saved model?"
        )
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def plot_training_curves(history: dict, save_path: str, title_prefix: str = "") -> None:
    """
    Plot training vs validation loss, and training vs validation accuracy,
    side by side, and save the figure to disk.

    `history` is expected to be a dict with these keys (lists, one value
    per epoch):
        train_loss, val_loss, train_acc, val_acc
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Loss plot ---
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title_prefix}Loss vs Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy plot ---
    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Validation Accuracy", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{title_prefix}Accuracy vs Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: {save_path}")


def save_history_json(history: dict, path: str) -> None:
    """Save the raw per-epoch numbers so experiments can be compared later."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def project_root() -> str:
    """
    Return the absolute path of the project root folder, computed
    relative to this file. This is what makes all paths in the project
    portable: nothing is hardcoded like 'C:/Users/me/project'.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
