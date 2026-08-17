"""Train and evaluate logistic regression on synthetic CTR data."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DataSplit, prepare_data
from .model import LogisticRegression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-cross", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lr"))
    return parser.parse_args()


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


def evaluate(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, clipped)),
        "ece": expected_calibration_error(labels, probabilities),
        "predicted_ctr": float(probabilities.mean()),
        "observed_ctr": float(labels.mean()),
    }


def train(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")

    set_seed(args.seed)
    data: DataSplit = prepare_data(
        samples=args.samples,
        seed=args.seed,
        include_cross=args.include_cross,
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

    print(f"features={data.feature_names}")
    print(f"samples={args.samples} overall_ctr={data.true_ctr:.4f}")

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
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )

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

    experiment_name = "with_cross" if args.include_cross else "baseline"
    output_dir = args.output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "weights.json").write_text(
        json.dumps(learned_parameters, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": data.feature_names,
            "mean": data.mean,
            "std": data.std,
            "metrics": metrics,
        },
        output_dir / "model.pt",
    )

    print("\ntest metrics")
    for name in ("auc", "log_loss", "ece", "predicted_ctr", "observed_ctr"):
        print(f"{name:>14}: {metrics[name]:.6f}")

    print("\nlearned standardized-feature weights")
    for name, weight in learned_parameters["weights"].items():
        print(f"{name:>14}: {weight:+.6f}")
    print(f"{'bias':>14}: {learned_parameters['bias']:+.6f}")
    print(f"\nsaved_to={output_dir}")


if __name__ == "__main__":
    train(parse_args())
