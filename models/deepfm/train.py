"""Train DeepFM branch ablations on the existing sparse CTR protocol."""

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
from torch.utils.data import DataLoader, TensorDataset

from models.lr.train import environment_info, evaluate, set_seed, split_summary, write_json
from models.sparse_ctr.train import predict

from .data import CATEGORICAL_FIELDS, NUMERIC_FIELDS, prepare_data
from .model import DeepFM, VARIANTS


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="deepfm")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deepfm"))
    return parser.parse_args(argv)


def train(args: argparse.Namespace) -> Path:
    if args.variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if args.samples < 100 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("samples must be >= 100; epochs and batch-size must be positive")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError("learning-rate must be finite and positive")
    if args.embedding_dim <= 0 or args.patience < 0:
        raise ValueError("embedding-dim must be positive and patience nonnegative")

    set_seed(args.seed)
    data = prepare_data(args.samples, args.seed)
    model = DeepFM(
        total_tokens=sum(data.vocab_sizes),
        categorical_fields=len(CATEGORICAL_FIELDS),
        numeric_fields=len(NUMERIC_FIELDS),
        embedding_dim=args.embedding_dim,
        variant=args.variant,
    )
    train_dataset = TensorDataset(
        torch.from_numpy(data.train_cat),
        torch.from_numpy(data.train_num),
        torch.from_numpy(data.train_y),
    )
    # Fixed independently of model initialization so variants see the same batch order.
    batch_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=batch_generator
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    created_at = datetime.now(timezone.utc)
    run_id = f"seed_{args.seed}_{created_at:%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"
    output_dir = args.output_dir / args.variant / run_id
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
        "data": {
            "generator": "models.sparse_ctr.data.generate_exposures",
            "split_method": "sequential_synthetic_iid",
            "categorical_fields": list(CATEGORICAL_FIELDS),
            "numeric_fields": list(NUMERIC_FIELDS),
            "vocabulary_fit": "train_only",
            "oov_local_id": 0,
            "padding_used": False,
            "vocab_sizes_including_oov": list(data.vocab_sizes),
            "field_offsets": list(data.offsets),
            "oov_counts": data.oov_counts,
            "numeric_mean": data.numeric_mean.tolist(),
            "numeric_std": data.numeric_std.tolist(),
            "splits": splits,
        },
        "model": {
            "first_order": True,
            "fm_branch": args.variant in ("fm_only", "deepfm"),
            "deep_branch": args.variant in ("deep_only", "deepfm"),
            "shared_embeddings": args.variant == "deepfm",
            "hidden_units": [64, 32] if args.variant != "fm_only" else [],
        },
        "training": {
            "optimizer": "Adam", "loss": "BCEWithLogitsLoss",
            "selection_metric": "validation_bce", "restore_best": True,
            "batch_order_seed": args.seed,
        },
        "metric_definitions": {
            "pr_auc": "average_precision", "ece": "10 equal-width probability bins",
            "classification_threshold": 0.5,
        },
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
        for batch_cat, batch_num, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_cat, batch_num)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(batch_y)
        train_loss = train_loss_sum / len(data.train_y)
        _, val_loss = predict(
            model, data.val_cat, data.val_num, data.val_y, args.batch_size
        )
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise ValueError("non-finite loss")
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
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if args.patience and stale_epochs >= args.patience:
            print(f"early_stopping: epoch={epoch}")
            break
    training_seconds = time.perf_counter() - started

    model.load_state_dict(best_state)
    test_probabilities, test_bce = predict(
        model, data.test_cat, data.test_num, data.test_y, args.batch_size
    )
    metrics = evaluate(data.test_y, test_probabilities)
    metrics.update({
        "bce_from_torch": float(test_bce),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(history),
        "stopped_early": len(history) < args.epochs,
        "training_seconds": training_seconds,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "splits": splits,
    })
    write_json(output_dir / "metrics.json", metrics)
    torch.save({
        "state_dict": model.state_dict(),
        "vocab_values": [torch.from_numpy(values.copy()) for values in data.vocab_values],
        "vocab_sizes": data.vocab_sizes,
        "offsets": data.offsets,
        "numeric_mean": torch.from_numpy(data.numeric_mean.copy()),
        "numeric_std": torch.from_numpy(data.numeric_std.copy()),
        "config": config,
        "metrics": metrics,
    }, output_dir / "model.pt")
    print(f"best_epoch={best_epoch} val_loss={best_val_loss:.6f}")
    print("test:", {key: metrics[key] for key in
                    ("auc", "pr_auc", "log_loss", "ece", "observed_ctr")})
    print(f"saved_to={output_dir}")
    return output_dir


if __name__ == "__main__":
    train(parse_args())
