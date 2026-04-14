"""Experiment 2 — Architecture Selection.

Trains S5, Mamba, Transformer, and xLSTM on Truflation CPI data
with the same forecasting objective and ~10M param budget.
Picks the winning architecture.

Usage:
    python scripts/experiment_02_architecture.py --arch all --horizon 90
    python scripts/experiment_02_architecture.py --arch s5 mamba --horizon 30

Run on Vast.ai with A100. Estimated time: ~90 min for all 4 architectures.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_categories
from src.dataset import TimeSeriesDataset, normalize_streams
from src.models import MODEL_REGISTRY
from src.trainer import train_model, evaluate
from src.metrics import evaluate_forecast


RESULTS_DIR = Path(__file__).parent.parent / "results" / "experiment_02"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Architecture configs targeting ~10M params each
ARCH_CONFIGS = {
    "transformer": {
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 6,
        "patch_size": 16,
        "dropout": 0.1,
    },
    "s5": {
        "d_model": 256,
        "state_dim": 64,
        "num_layers": 6,
        "dropout": 0.1,
    },
    "mamba": {
        "d_model": 256,
        "state_dim": 64,
        "num_layers": 6,
        "dropout": 0.1,
    },
    "xlstm": {
        "d_model": 256,
        "hidden_dim": 256,
        "num_layers": 4,
        "dropout": 0.1,
    },
}


def get_data(horizon: int, context_len: int = 512):
    """Load and prepare Truflation data for training."""
    print("Loading Truflation data...")
    cat_df = load_categories(frozen=True)

    # Use all Index columns (84 streams) — not just the 32 from Exp 1
    index_cols = sorted([c for c in cat_df.columns if 'Index' in c or 'index' in c])
    # Remove BEA year_ago and yoy columns that were caught by the filter
    index_cols = [c for c in index_cols if '_year_ago' not in c and '_yoy' not in c]

    print(f"Using {len(index_cols)} index streams")

    # Extract data matrix
    data = cat_df[index_cols].values.astype(np.float64)
    dates = cat_df["date"].values

    # Handle NaN with forward fill then backward fill
    df_clean = pd.DataFrame(data, columns=index_cols).ffill().bfill()
    data = df_clean.values

    # Normalize per-stream
    data_norm, means, stds = normalize_streams(data)

    # Train/val/test split by date
    # Train: 2010-2022, Val: 2023, Test: 2024-2026
    train_end = np.searchsorted(dates, np.datetime64("2023-01-01"))
    val_end = np.searchsorted(dates, np.datetime64("2024-01-01"))

    train_data = data_norm[:train_end]
    val_data = data_norm[:val_end]  # Val includes train context for sliding windows
    test_data = data_norm  # Test includes all data for context

    print(f"Train: {train_end} days, Val: {val_end - train_end} days, Test: {len(data) - val_end} days")
    print(f"Streams: {data.shape[1]}, Context: {context_len}, Horizon: {horizon}")

    # Create datasets
    train_ds = TimeSeriesDataset(train_data, context_len, horizon, stride=7)
    val_ds = TimeSeriesDataset(val_data[train_end - context_len:], context_len, horizon, stride=7)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    return train_ds, val_ds, data_norm, means, stds, dates, val_end, index_cols


def run_test_eval(
    model,
    data_norm,
    means,
    stds,
    dates,
    test_start_idx,
    index_cols,
    context_len,
    horizons,
    device,
):
    """Run proper test evaluation matching Experiment 1 protocol."""
    model.eval()
    num_streams = len(index_cols)
    results = {}

    for h in horizons:
        # Use context ending at test_start_idx, predict h days forward
        context = data_norm[test_start_idx - context_len : test_start_idx]
        actual_norm = data_norm[test_start_idx : test_start_idx + h]

        if len(actual_norm) < h:
            continue

        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # For models trained with different horizon, we may need to handle this
            pred_norm = model(context_tensor).cpu().numpy()[0]

        # Denormalize
        pred = pred_norm[:h] * stds + means
        actual = actual_norm[:h] * stds + means

        # Per-stream metrics
        stream_maes = []
        stream_mases = []
        stream_dirs = []

        train_data_raw = data_norm[:test_start_idx] * stds + means

        for s in range(num_streams):
            stream_pred = pred[:, s]
            stream_actual = actual[:, s]
            stream_train = train_data_raw[:, s]

            mae = np.mean(np.abs(stream_pred - stream_actual))

            # MASE
            naive_errors = np.abs(stream_train[365:] - stream_train[:-365])
            naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
            mase = mae / naive_mae if naive_mae > 0 else float("inf")

            # Direction
            if h > 1:
                pred_dir = np.sign(np.diff(stream_pred))
                actual_dir = np.sign(np.diff(stream_actual))
                dir_acc = np.mean(pred_dir == actual_dir)
            else:
                dir_acc = 0.5

            stream_maes.append(mae)
            stream_mases.append(mase)
            stream_dirs.append(dir_acc)

        results[f"h{h}"] = {
            "mae": np.mean(stream_maes),
            "mase": np.mean(stream_mases),
            "directional_accuracy": np.mean(stream_dirs),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Architecture Selection")
    parser.add_argument(
        "--arch",
        nargs="+",
        default=["all"],
        choices=["transformer", "s5", "mamba", "xlstm", "all"],
    )
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if "all" in args.arch:
        args.arch = ["transformer", "s5", "mamba", "xlstm"]

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_ds, val_ds, data_norm, means, stds, dates, test_start, index_cols = get_data(
        args.horizon, args.context_len
    )
    num_streams = len(index_cols)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_results = {}

    for arch_name in args.arch:
        print(f"\n{'='*65}")
        print(f"  Architecture: {arch_name.upper()}")
        print(f"{'='*65}")

        config = ARCH_CONFIGS[arch_name].copy()
        model_cls = MODEL_REGISTRY[arch_name]

        model = model_cls(
            num_streams=num_streams,
            horizon=args.horizon,
            **config,
        )

        print(f"Parameters: {model.count_params():,}")

        # Train
        train_result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            model_name=arch_name,
        )

        # Test evaluation at multiple horizons
        test_metrics = run_test_eval(
            model=model,
            data_norm=data_norm,
            means=means,
            stds=stds,
            dates=dates,
            test_start_idx=test_start,
            index_cols=index_cols,
            context_len=args.context_len,
            horizons=[7, 30, 90],
            device=args.device,
        )

        all_results[arch_name] = {
            "config": config,
            "num_params": model.count_params(),
            "training_time": train_result["training_time"],
            "best_val_metrics": train_result["best_metrics"],
            "test_metrics": test_metrics,
        }

        # Save checkpoint
        ckpt_path = RESULTS_DIR / f"{arch_name}_best.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # Summary
    print(f"\n{'='*65}")
    print("  EXPERIMENT 2 — ARCHITECTURE COMPARISON")
    print(f"{'='*65}")
    print(f"\n{'Architecture':<15} {'Params':>10} {'Time':>8} {'7d MAE':>10} {'30d MAE':>10} {'90d MAE':>10} {'90d Dir%':>10}")
    print("-" * 80)

    for name, res in all_results.items():
        tm = res["test_metrics"]
        print(
            f"{name:<15} {res['num_params']:>10,} {res['training_time']:>7.0f}s "
            f"{tm.get('h7', {}).get('mae', -1):>10.4f} "
            f"{tm.get('h30', {}).get('mae', -1):>10.4f} "
            f"{tm.get('h90', {}).get('mae', -1):>10.4f} "
            f"{tm.get('h90', {}).get('directional_accuracy', -1):>9.2%}"
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"architecture_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
