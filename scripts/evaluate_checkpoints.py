"""Evaluate saved Experiment 2 checkpoints against Experiment 1 baselines.

Loads each architecture's best checkpoint and runs the same evaluation
protocol as Experiment 1: MAE, MASE, directional accuracy at 7/30/90 days
on 2024-2026 test data.

Usage:
    python scripts/evaluate_checkpoints.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_categories
from src.dataset import normalize_streams
from src.models import MODEL_REGISTRY


RESULTS_DIR = Path(__file__).parent.parent / "results" / "experiment_02"
HORIZONS = [7, 30, 90]

# Must match experiment_02_architecture.py configs
ARCH_CONFIGS = {
    "transformer": {
        "d_model": 256, "num_heads": 8, "num_layers": 6,
        "patch_size": 16, "dropout": 0.0,
    },
    "s5": {
        "d_model": 256, "state_dim": 64, "num_layers": 6, "dropout": 0.0,
    },
    "mamba": {
        "d_model": 256, "state_dim": 64, "num_layers": 6, "dropout": 0.0,
    },
    "xlstm": {
        "d_model": 256, "hidden_dim": 256, "num_layers": 4, "dropout": 0.0,
    },
}


def load_data():
    """Load and prepare data matching Experiment 2 protocol."""
    cat_df = load_categories(frozen=True)

    index_cols = sorted([c for c in cat_df.columns if 'Index' in c or 'index' in c])
    index_cols = [c for c in index_cols if '_year_ago' not in c and '_yoy' not in c]

    data = cat_df[index_cols].values.astype(np.float64)
    dates = cat_df["date"].values

    df_clean = pd.DataFrame(data, columns=index_cols).ffill().bfill()
    data = df_clean.values

    # Normalize using TRAINING data stats only (no leakage)
    train_end = np.searchsorted(dates, np.datetime64("2023-01-01"))
    train_data = data[:train_end]
    means = np.mean(train_data, axis=0)
    stds = np.std(train_data, axis=0)
    stds = np.where(stds == 0, 1.0, stds)

    data_norm = (data - means) / stds

    test_start = np.searchsorted(dates, np.datetime64("2024-01-01"))

    return data_norm, data, means, stds, dates, test_start, index_cols


def evaluate_checkpoint(
    arch_name, model, data_norm, data_raw, means, stds, test_start, context_len=512
):
    """Evaluate a model checkpoint at multiple horizons."""
    device = next(model.parameters()).device
    model.eval()
    results = {}

    for h in HORIZONS:
        context_norm = data_norm[test_start - context_len : test_start]
        actual_norm = data_norm[test_start : test_start + h]

        if len(actual_norm) < h:
            continue

        context_tensor = torch.tensor(
            context_norm, dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = model(context_tensor).cpu().numpy()[0][:h]

        # Denormalize
        pred = pred_norm * stds + means
        actual = actual_norm * stds + means

        # Per-stream metrics
        num_streams = pred.shape[1]
        train_raw = data_raw[:test_start]
        stream_maes, stream_mases, stream_dirs = [], [], []

        for s in range(num_streams):
            sp, sa, st = pred[:, s], actual[:, s], train_raw[:, s]

            mae = np.mean(np.abs(sp - sa))

            naive_err = np.abs(st[365:] - st[:-365])
            naive_mae = np.mean(naive_err) if len(naive_err) > 0 else 1.0
            mase = mae / naive_mae if naive_mae > 0 else float("inf")

            if h > 1:
                dir_acc = np.mean(np.sign(np.diff(sp)) == np.sign(np.diff(sa)))
            else:
                dir_acc = 0.5

            stream_maes.append(mae)
            stream_mases.append(mase)
            stream_dirs.append(dir_acc)

        results[h] = {
            "mae": float(np.mean(stream_maes)),
            "mase": float(np.mean(stream_mases)),
            "directional_accuracy": float(np.mean(stream_dirs)),
        }

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading data...")
    data_norm, data_raw, means, stds, dates, test_start, index_cols = load_data()
    num_streams = len(index_cols)
    print(f"Streams: {num_streams}, Test start idx: {test_start}")

    all_results = {}

    for arch_name in ["transformer", "s5", "mamba", "xlstm"]:
        ckpt_path = RESULTS_DIR / f"{arch_name}_best.pt"
        if not ckpt_path.exists():
            print(f"\n{arch_name}: no checkpoint found, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"  Evaluating: {arch_name.upper()}")
        print(f"{'='*50}")

        config = ARCH_CONFIGS[arch_name]
        model = MODEL_REGISTRY[arch_name](
            num_streams=num_streams, horizon=90, **config
        ).to(device)

        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded {ckpt_path} ({model.count_params():,} params)")

        results = evaluate_checkpoint(
            arch_name, model, data_norm, data_raw, means, stds, test_start
        )
        all_results[arch_name] = results

        for h, m in results.items():
            print(f"  {h:3d}d — MAE: {m['mae']:.4f}  MASE: {m['mase']:.4f}  Dir: {m['directional_accuracy']:.2%}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("  EXPERIMENT 2 — TEST RESULTS vs EXPERIMENT 1 BASELINES")
    print(f"{'='*80}")

    # Exp 1 baselines for comparison
    baselines = {
        "TiRex (35M)":        {7: 1.07, 30: 1.41, 90: 1.72},
        "Chronos-T5 (46M)":   {7: 1.05, 30: 1.52, 90: 2.06},
        "EMA":                {7: 1.09, 30: 1.37, 90: 1.90},
        "TimesFM (200M)":     {7: 1.09, 30: 1.70, 90: 2.39},
        "Toto (151M)":        {7: 1.15, 30: 1.42, 90: 2.05},
        "Moirai (14M)":       {7: 1.15, 30: 1.66, 90: 2.09},
    }

    print(f"\n{'Model':<25} {'7d MAE':>10} {'30d MAE':>10} {'90d MAE':>10} {'90d Dir%':>10}")
    print("-" * 70)

    for name, res in all_results.items():
        params = ARCH_CONFIGS[name]
        p = sum(p.numel() for p in MODEL_REGISTRY[name](
            num_streams=num_streams, horizon=90, **params
        ).parameters())
        label = f"Thales-{name} ({p/1e6:.1f}M)"
        print(f"{label:<25} {res.get(7,{}).get('mae',0):>10.4f} "
              f"{res.get(30,{}).get('mae',0):>10.4f} "
              f"{res.get(90,{}).get('mae',0):>10.4f} "
              f"{res.get(90,{}).get('directional_accuracy',0):>9.2%}")

    print("-" * 70)
    for name, maes in baselines.items():
        print(f"{name:<25} {maes[7]:>10.4f} {maes[30]:>10.4f} {maes[90]:>10.4f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"checkpoint_eval_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
