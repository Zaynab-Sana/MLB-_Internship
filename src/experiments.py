"""
experiments.py
--------------
Day-3 deliverables that build on train.py:

1. compare_optimizers() - trains the SAME SimpleCNN architecture once
   with SGD and once with Adam, keeping every other setting identical,
   then prints/saves a side-by-side comparison table.

2. run_hyperparameter_experiments() - runs a small, deliberately
   limited set of controlled experiments varying one setting at a time
   (learning rate, batch size, number of filters, dropout), and reports
   validation/test accuracy for each in a results table.

Both functions reuse train_model() and evaluate_model() from
train.py / evaluate.py -- nothing about the model or training loop is
duplicated here, this file only orchestrates multiple runs and
tabulates the results.

Run it like this:
    python -m src.experiments --mode optimizer_comparison
    python -m src.experiments --mode hyperparameters

NOTE: Each call to train_model() launches a full training run. Keep
--epochs modest here (a handful of experiments x many epochs each adds
up quickly on a laptop).
"""

import argparse

from src.train import train_model
from src.evaluate import evaluate_model


def _base_args(**overrides):
    """
    Build an argparse.Namespace with train.py's defaults, then override
    specific fields. This lets us launch several training runs
    programmatically without duplicating the CLI parsing logic.
    """
    # argparse.parse_args() reads sys.argv by default, which we don't want
    # here since this is a programmatic sweep. Build a fresh Namespace by
    # hand, mirroring train.py's parser, but with an empty argument list.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--num_filters", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def compare_optimizers(epochs: int = 8):
    """
    Trains the identical SimpleCNN architecture with SGD and with Adam.
    Everything else (seed, batch size, filters, dropout, epochs) is kept
    the same so the comparison isolates the effect of the optimizer.
    """
    results = {}

    for opt_name, lr in [("sgd", 0.01), ("adam", 0.001)]:
        args = _base_args(optimizer=opt_name, lr=lr, epochs=epochs, run_name=opt_name)
        history, model_path = train_model(args)
        test_metrics = evaluate_model(model_path=model_path, seed=args.seed, batch_size=args.batch_size)
        results[opt_name] = {
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1],
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "best_val_acc": history["best_val_acc"],
            "training_time_seconds": history["training_time_seconds"],
            "test_accuracy": test_metrics["accuracy"] * 100,
        }

    print("\n=== SGD vs Adam Comparison ===")
    header = f"{'Metric':<22}{'SGD':>12}{'Adam':>12}"
    print(header)
    print("-" * len(header))
    for metric in ["final_train_acc", "final_val_acc", "best_val_acc", "test_accuracy",
                   "final_train_loss", "final_val_loss", "training_time_seconds"]:
        row = f"{metric:<22}{results['sgd'][metric]:>12.3f}{results['adam'][metric]:>12.3f}"
        print(row)

    winner = "Adam" if results["adam"]["test_accuracy"] >= results["sgd"]["test_accuracy"] else "SGD"
    print(
        f"\n{winner} achieved the higher test accuracy in this run. In general, Adam tends to "
        f"converge faster and needs less learning-rate tuning because it adapts the step size "
        f"per parameter, which often helps on small/medium datasets and short training runs. "
        f"Plain SGD (even with momentum) can match or beat Adam given enough epochs and a "
        f"well-tuned learning rate/schedule, and sometimes generalizes slightly better, but it "
        f"is more sensitive to the learning rate choice. Results can vary between runs and "
        f"datasets, so treat this as a starting point, not a universal rule."
    )
    return results


def run_hyperparameter_experiments(epochs: int = 6):
    """
    A small, deliberately limited set of controlled experiments, changing
    one hyperparameter at a time from a fixed baseline. This is meant to
    illustrate the EFFECT of each setting, not to exhaustively search for
    the best possible combination.
    """
    baseline = dict(lr=0.001, batch_size=32, num_filters=32, dropout=0.5, optimizer="adam")

    experiment_configs = [
        ("baseline",            {}),
        ("higher_lr",           {"lr": 0.005}),
        ("lower_lr",            {"lr": 0.0005}),
        ("larger_batch",        {"batch_size": 64}),
        ("more_filters",        {"num_filters": 64}),
        ("higher_dropout",      {"dropout": 0.7}),
    ]

    rows = []
    for exp_name, override in experiment_configs:
        config = {**baseline, **override}
        args = _base_args(
            lr=config["lr"], batch_size=config["batch_size"], num_filters=config["num_filters"],
            dropout=config["dropout"], optimizer=config["optimizer"], epochs=epochs, run_name=exp_name,
        )
        history, model_path = train_model(args)
        test_metrics = evaluate_model(model_path=model_path, seed=args.seed, batch_size=args.batch_size)

        rows.append({
            "experiment": exp_name,
            "lr": config["lr"],
            "batch_size": config["batch_size"],
            "num_filters": config["num_filters"],
            "dropout": config["dropout"],
            "val_accuracy": history["best_val_acc"],
            "test_accuracy": test_metrics["accuracy"] * 100,
        })

    print("\n=== Hyperparameter Experiment Results ===")
    header = f"{'Experiment':<16}{'LR':>8}{'Batch':>8}{'Filters':>9}{'Dropout':>9}{'ValAcc%':>10}{'TestAcc%':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['experiment']:<16}{row['lr']:>8}{row['batch_size']:>8}{row['num_filters']:>9}"
              f"{row['dropout']:>9}{row['val_accuracy']:>10.2f}{row['test_accuracy']:>10.2f}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["optimizer_comparison", "hyperparameters"],
                         required=True)
    parser.add_argument("--epochs", type=int, default=8)
    cli_args = parser.parse_args()

    if cli_args.mode == "optimizer_comparison":
        compare_optimizers(epochs=cli_args.epochs)
    else:
        run_hyperparameter_experiments(epochs=cli_args.epochs)
