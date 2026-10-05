"""Train a second-order FM on the original LR CTR data."""

import argparse
import copy
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence
from uuid import uuid4

import torch
from torch import nn

from models.lr.train import (
    environment_info,
    evaluate,
    make_loader,
    predict,
    set_seed,
    split_summary,
    write_json,
)

from .data import prepare_data
from .model import FactorizationMachine


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fm"))
    return parser.parse_args(argv)


def train(args: argparse.Namespace) -> Path:
    if args.epochs <= 0 or args.batch_size <= 0 or args.embedding_dim <= 0:
        raise ValueError("epochs, batch-size, and embedding-dim must be positive")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError("learning-rate must be finite and positive")
    if args.patience < 0:
        raise ValueError("patience must be nonnegative")

    set_seed(args.seed)
    data = prepare_data(samples=args.samples, seed=args.seed)
    train_loader = make_loader(data.train_x, data.train_y, args.batch_size, shuffle=True)
    model = FactorizationMachine(len(data.feature_names), args.embedding_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    created_at = datetime.now(timezone.utc)
    run_id = f"seed_{args.seed}_{created_at:%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    splits = {
        "train": split_summary(data.train_y),
        "validation": split_summary(data.val_y),
        "test": split_summary(data.test_y),
    }
    config = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "arguments": {**vars(args), "output_dir": str(args.output_dir)},
        "feature_names": list(data.feature_names),
        "data": {"split_method": "sequential_synthetic_iid", "splits": splits,
                 "manual_cross": False, "source": "models.lr.data.prepare_data"},
        "training": {"optimizer": "Adam", "loss": "BCEWithLogitsLoss",
                     "selection_metric": "validation_bce", "restore_best": True},
        "metric_definitions": {"pr_auc": "average_precision",
                               "ece": "10 equal-width probability bins",
                               "classification_threshold": 0.5,
                               "single_class_ranking_metrics": "null (not comparable)"},
        "environment": environment_info(),
    }
    write_json(output_dir / "config.json", config)

    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

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
        _, val_loss = predict(model, data.val_x, data.val_y, args.batch_size)
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise ValueError("non-finite loss; check learning rate and input data")
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "is_best": improved})
        write_json(output_dir / "history.json", history)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if args.patience and stale_epochs >= args.patience:
            print(f"early_stopping: epoch={epoch} patience={args.patience}")
            break

    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    test_probabilities, test_bce = predict(
        model, data.test_x, data.test_y, args.batch_size
    )
    metrics = evaluate(data.test_y, test_probabilities)
    metrics.update({
        "bce_from_torch": float(test_bce),
        "samples": args.samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "embedding_dim": args.embedding_dim,
        "seed": args.seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stopped_early": len(history) < args.epochs,
        "training_seconds": training_seconds,
        "splits": splits,
    })
    write_json(output_dir / "metrics.json", metrics)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_names": data.feature_names,
        "mean": torch.from_numpy(data.mean.copy()),
        "std": torch.from_numpy(data.std.copy()),
        "metrics": metrics,
        "config": config,
    }, output_dir / "model.pt")
    print(f"best_epoch={best_epoch} val_loss={best_val_loss:.6f}")
    print("test:", {name: metrics[name] for name in
                    ("auc", "pr_auc", "log_loss", "ece", "predicted_ctr", "observed_ctr")})
    print(f"saved_to={output_dir}")
    return output_dir


if __name__ == "__main__":
    train(parse_args())
