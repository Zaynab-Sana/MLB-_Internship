"""
train.py
--------
Trains SimpleCNN on TrashNet and saves the best model (by validation
accuracy) to disk. Also saves training curves and a JSON history file
so results from different runs (e.g. SGD vs Adam, or different
hyperparameters) can be compared later.

Run it like this:
    python -m src.train
    python -m src.train --optimizer adam --lr 0.001 --epochs 15
    python -m src.train --optimizer sgd --lr 0.01 --momentum 0.9 --epochs 15

--------------------------------------------------------------------
WHY CrossEntropyLoss
--------------------------------------------------------------------
CrossEntropyLoss is the standard loss for multi-class, single-label
classification (each image belongs to exactly one of the 6 classes).
It internally combines LogSoftmax + Negative Log-Likelihood in one
numerically stable step. It expects:
    - model output: raw, un-normalized scores ("logits"), shape (batch, 6)
    - target: the correct class index as an integer, shape (batch,)

WHY WE DO NOT APPLY SOFTMAX MANUALLY BEFORE CrossEntropyLoss
PyTorch's CrossEntropyLoss already applies Softmax internally (via
LogSoftmax) before computing the loss. If we also applied Softmax
ourselves inside the model, the loss would effectively apply it TWICE,
which distorts the gradients and usually hurts training. That's why
SimpleCNN.forward() returns raw logits, and Softmax is only applied
later, manually, at INFERENCE time in predict.py to turn the logits
into human-readable confidence percentages.

--------------------------------------------------------------------
WHY SGD AND WHAT ITS PARAMETERS MEAN
--------------------------------------------------------------------
- Learning rate (lr): how big a step the optimizer takes when updating
  weights after each batch. Too high -> training diverges/oscillates.
  Too low -> training is painfully slow or gets stuck. 0.01 is a
  reasonable starting point for SGD on a small CNN like this.
- Momentum: keeps a running average of past gradients so the optimizer
  keeps moving in a consistent direction instead of zig-zagging,
  similar to a ball rolling downhill picking up speed. 0.9 is a common
  default.
- Parameters: the learnable numbers of the network (all the conv
  filters and fully-connected weights) -- what the optimizer updates.
- Optimization: the general process of repeatedly nudging the
  parameters in the direction that reduces the loss.

Adam is compared against SGD in the Day-3 experiments (see
compare_optimizers.py-equivalent workflow described in the README);
this file supports either via --optimizer.
--------------------------------------------------------------------
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device, save_checkpoint, plot_training_curves, save_history_json, project_root


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimpleCNN on TrashNet")
    parser.add_argument("--dataset_dir", type=str, default=None,
                         help="Path to dataset folder (default: <project_root>/dataset)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None,
                         help="Learning rate. Defaults: 0.01 for SGD, 0.001 for Adam.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Only used by SGD.")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--num_filters", type=int, default=32,
                         help="Number of filters in the first conv layer.")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None,
                         help="Name used for saved model/graph files, e.g. 'sgd_run1'. "
                              "Defaults to the optimizer name.")
    return parser.parse_args()


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    """
    Runs one full pass over `loader`.
    If train=True: does forward + backward + optimizer step (learning).
    If train=False: only does forward pass (used for validation, no learning).

    Returns: (average_loss, accuracy_percent)
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # torch.set_grad_enabled toggles gradient tracking. We disable it
    # during validation to save memory and computation, since we don't
    # need gradients when we're not updating weights.
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()  # clear gradients from the previous batch

            outputs = model(images)            # forward pass -> raw logits, shape (batch, 6)
            loss = criterion(outputs, labels)  # compare predictions to true labels

            if train:
                loss.backward()   # backpropagation: compute gradient of loss w.r.t. every weight
                optimizer.step()  # update weights using those gradients

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)      # pick the class with the highest logit
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(args=None):
    """
    Main training entry point. Can be called with a parsed argparse
    Namespace (for programmatic experiments, e.g. hyperparameter sweeps)
    or run directly from the command line.
    """
    if args is None:
        args = parse_args()

    set_seed(args.seed)
    device = get_device()

    # Sensible default learning rate per optimizer, if the user didn't pick one.
    lr = args.lr if args.lr is not None else (0.01 if args.optimizer == "sgd" else 0.001)
    run_name = args.run_name or args.optimizer

    print(f"\n=== Training run: '{run_name}' ===")
    print(f"optimizer={args.optimizer}, lr={lr}, momentum={args.momentum if args.optimizer == 'sgd' else 'N/A'}, "
          f"epochs={args.epochs}, batch_size={args.batch_size}, num_filters={args.num_filters}, "
          f"dropout={args.dropout}\n")

    train_loader, val_loader, _, classes = get_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(num_classes=len(classes), num_filters=args.num_filters,
                         dropout=args.dropout, device=device)

    # Multiclass classification loss. See module docstring for why.
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(project_root(), "models", f"best_model_{run_name}.pth")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        # Save the model only when it beats the best validation accuracy
        # seen so far. This protects against overfitting in later epochs:
        # we keep the checkpoint from the point where the model
        # generalized best, not necessarily the very last epoch.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_model_path, extra_info={
                "classes": classes,
                "num_filters": args.num_filters,
                "dropout": args.dropout,
                "epoch": epoch,
                "val_acc": val_acc,
            })

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time:.1f} seconds. Best validation accuracy: {best_val_acc:.2f}%")

    # Save curves + raw numbers for later comparison (SGD vs Adam, hyperparameter table, etc.)
    graphs_dir = os.path.join(project_root(), "outputs", "graphs")
    plot_training_curves(history, os.path.join(graphs_dir, f"curves_{run_name}.png"),
                          title_prefix=f"{run_name.upper()} - ")
    save_history_json(history, os.path.join(graphs_dir, f"history_{run_name}.json"))

    history["training_time_seconds"] = total_time
    history["best_val_acc"] = best_val_acc
    return history, best_model_path


if __name__ == "__main__":
    train_model()
