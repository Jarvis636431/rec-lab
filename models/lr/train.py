"""Train and evaluate logistic regression on synthetic CTR data."""

import argparse
import copy
import json
import math
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DataSplit, prepare_data
from .model import LogisticRegression


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience", type=int, default=0,
        help="stop after this many epochs without validation improvement; 0 disables stopping",
    )
    parser.add_argument("--include-cross", action="store_true")
    parser.add_argument(
        "--base-ctr",
        type=float,
        default=None,
        help=(
            "reference-impression CTR before feature effects; "
            "omit to preserve the original data generator"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lr"))
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(
    model: LogisticRegression,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    loader = make_loader(features, labels, batch_size, shuffle=False)

    probabilities = []
    total_loss = 0.0
    total_examples = 0
    for batch_x, batch_y in loader:
        logits = model(batch_x)
        total_loss += criterion(logits, batch_y).item()
        total_examples += batch_y.numel()
        probabilities.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probabilities), total_loss / total_examples


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Measure the weighted gap between predicted and observed click rates."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        predicted_ctr = float(probabilities[mask].mean())
        observed_ctr = float(labels[mask].mean())
        error += float(mask.sum()) / total * abs(predicted_ctr - observed_ctr)
    return error


def evaluate(
    labels: np.ndarray, probabilities: np.ndarray,
) -> Dict[str, Optional[float]]:
    if labels.ndim != 1 or probabilities.shape != labels.shape or labels.size == 0:
        raise ValueError("labels and probabilities must be nonempty matching 1-D arrays")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("probabilities must be finite and between 0 and 1")
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    predictions = (probabilities >= 0.5).astype(np.float32)
    has_both_classes = np.unique(labels).size == 2
    return {
        "auc": float(roc_auc_score(labels, probabilities)) if has_both_classes else None,
        # Keep the existing key; its precise definition is Average Precision (AP).
        "pr_auc": float(average_precision_score(labels, probabilities))
        if has_both_classes else None,
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "ece": expected_calibration_error(labels, probabilities),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "predicted_ctr": float(probabilities.mean()),
        "observed_ctr": float(labels.mean()),
    }


def evaluate_always_negative(labels: np.ndarray) -> Dict[str, Optional[float]]:
    """Evaluate the misleading majority-class baseline for low-CTR data."""

    probabilities = np.zeros_like(labels, dtype=np.float32)
    return evaluate(labels, probabilities)


def experiment_name(include_cross: bool, base_ctr: Optional[float]) -> str:
    """Build an experiment group name; each run gets its own subdirectory."""

    name = "with_cross" if include_cross else "baseline"
    if base_ctr is not None:
        ctr_label = format(base_ctr, ".8g").replace(".", "p")
        name = f"{name}_base_ctr_{ctr_label}"
    return name


def split_summary(labels: np.ndarray) -> Dict[str, object]:
    return {
        "samples": int(labels.size),
        "positives": int(labels.sum()),
        "negatives": int(labels.size - labels.sum()),
        "ctr": float(labels.mean()),
        "ranking_metrics_defined": bool(np.unique(labels).size == 2),
    }


def environment_info() -> Dict[str, object]:
    """Record the runtime and repository state, including runs outside a checkout."""
    repo = Path(__file__).resolve().parents[2]
    git_commit, git_dirty = None, None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": str(torch.__version__), "scikit_learn": sklearn.__version__,
        "device": "cpu", "torch_num_threads": torch.get_num_threads(),
        "git_commit": git_commit, "git_dirty": git_dirty,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def train(args: argparse.Namespace) -> Path:
    if (args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0
            or not math.isfinite(args.learning_rate)):
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    if args.patience < 0:
        raise ValueError("patience must be nonnegative")

    set_seed(args.seed)
    data: DataSplit = prepare_data(
        samples=args.samples,
        seed=args.seed,
        include_cross=args.include_cross,
        base_ctr=args.base_ctr,
    )
    train_loader = make_loader(
        data.train_x,
        data.train_y,
        args.batch_size,
        shuffle=True,
    )

    model = LogisticRegression(num_features=len(data.feature_names))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    created_at = datetime.now(timezone.utc)
    run_id = f"seed_{args.seed}_{created_at:%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"
    output_dir = args.output_dir / experiment_name(args.include_cross, args.base_ctr) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    splits = {
        "train": split_summary(data.train_y),
        "validation": split_summary(data.val_y),
        "test": split_summary(data.test_y),
    }
    config = {
        "schema_version": 1, "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "arguments": {**vars(args), "output_dir": str(args.output_dir)},
        "feature_names": list(data.feature_names),
        "data": {"split_method": "sequential_synthetic_iid", "splits": splits},
        "training": {"optimizer": "SGD", "loss": "BCEWithLogitsLoss",
                     "selection_metric": "validation_bce", "restore_best": True},
        "metric_definitions": {
            "pr_auc": "average_precision", "ece": "10 equal-width probability bins",
            "classification_threshold": 0.5,
            "single_class_ranking_metrics": "null (not comparable)",
        },
        "environment": environment_info(),
    }
    write_json(output_dir / "config.json", config)
    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    print(f"features={data.feature_names}")
    print(f"samples={args.samples} overall_ctr={data.true_ctr:.4f}")
    for name, summary in splits.items():
        print(f"{name}: {summary}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_y.numel()
            train_examples += batch_y.numel()

        train_loss = train_loss_sum / train_examples
        _, val_loss = predict(
            model,
            data.val_x,
            data.val_y,
            args.batch_size,
        )
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise ValueError("non-finite loss; check the learning rate and input data")
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "is_best": improved,
        })
        write_json(output_dir / "history.json", history)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )
        if args.patience and stale_epochs >= args.patience:
            print(f"early_stopping: epoch={epoch} patience={args.patience}")
            break

    # Select only by validation BCE. Test predictions happen once, after restoration.
    model.load_state_dict(best_state)
    print(f"restored_best_epoch={best_epoch} best_val_loss={best_val_loss:.6f}")
    test_probabilities, test_bce = predict(
        model,
        data.test_x,
        data.test_y,
        args.batch_size,
    )
    metrics = evaluate(data.test_y, test_probabilities)
    metrics.update(
        {
            "bce_from_torch": float(test_bce),
            "samples": args.samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "include_cross": args.include_cross,
            "base_ctr": args.base_ctr,
            "patience": args.patience,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "stopped_early": len(history) < args.epochs,
            "splits": splits,
        }
    )
    always_negative_metrics = evaluate_always_negative(data.test_y)
    metrics.update(
        {
            f"always_negative_{name}": value
            for name, value in always_negative_metrics.items()
        }
    )

    weights = model.linear.weight.detach().squeeze(0).cpu().numpy()
    learned_parameters = {
        "bias": float(model.linear.bias.detach().item()),
        "weights": {
            name: float(weight)
            for name, weight in zip(data.feature_names, weights)
        },
    }

    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "weights.json", learned_parameters)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": data.feature_names,
            # Tensors allow weights_only=True when reloading with modern PyTorch.
            "mean": torch.from_numpy(data.mean.copy()),
            "std": torch.from_numpy(data.std.copy()),
            "metrics": metrics,
            "config": config,
        },
        output_dir / "model.pt",
    )

    print("\ntest metrics")
    for name in (
        "auc",
        "pr_auc",
        "log_loss",
        "ece",
        "accuracy",
        "precision",
        "recall",
        "predicted_ctr",
        "observed_ctr",
    ):
        value = metrics[name]
        print(f"{name:>14}: {'undefined (single-class test set)' if value is None else f'{value:.6f}'}")

    print("\nalways-negative baseline")
    for name in ("accuracy", "precision", "recall", "auc", "pr_auc"):
        value = always_negative_metrics[name]
        print(f"{name:>14}: {'undefined (single-class test set)' if value is None else f'{value:.6f}'}")

    print("\nlearned standardized-feature weights")
    for name, weight in learned_parameters["weights"].items():
        print(f"{name:>14}: {weight:+.6f}")
    print(f"{'bias':>14}: {learned_parameters['bias']:+.6f}")
    print(f"\nsaved_to={output_dir}")
    return output_dir


if __name__ == "__main__":
    train(parse_args())
