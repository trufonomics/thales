"""Re-evaluate ALL models with corrected horizon-level directional accuracy.

Old metric: sign(diff(pred)) == sign(diff(actual)) on consecutive daily steps
  → Broken because CPI is flat 57.5% of days → all models get ~21%

New metric: Is predicted END value higher or lower than START value?
  → This is what clients care about: "will CPI be up or down in 90 days?"

Runs locally on CPU. No GPU needed.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_categories, extract_index_series, train_test_split_temporal


RESULTS_DIR = Path(__file__).parent.parent / "results"


def horizon_direction_accuracy(predicted_end, predicted_start, actual_end, actual_start):
    """Is the predicted direction (up/down) correct over the full horizon?

    For each stream: does sign(pred_end - pred_start) == sign(actual_end - actual_start)?
    Excludes streams where actual didn't move (flat).
    """
    pred_direction = np.sign(predicted_end - predicted_start)
    actual_direction = np.sign(actual_end - actual_start)

    # Only score streams that actually moved
    moved = actual_direction != 0
    if moved.sum() == 0:
        return 0.0, 0

    correct = (pred_direction[moved] == actual_direction[moved]).mean()
    return float(correct), int(moved.sum())


def eval_experiment1_baselines():
    """Re-evaluate Experiment 1 baselines from saved raw JSON results."""
    print("=" * 70)
    print("  EXPERIMENT 1 — CORRECTED DIRECTIONAL ACCURACY")
    print("=" * 70)

    # Load actual data
    cat_df = load_categories(frozen=True)
    idx_df = extract_index_series(cat_df)
    train_df, test_df = train_test_split_temporal(idx_df, "2024-01-01")
    streams = [c for c in idx_df.columns if c != "date"]

    # Find the raw results JSON from Exp 1 (Chronos + naive)
    raw_files = sorted(RESULTS_DIR.glob("experiment_01/baseline_raw_*.json"))

    all_model_results = {}

    for raw_file in raw_files:
        with open(raw_file) as f:
            raw = json.load(f)

        for model_name, model_data in raw.items():
            if model_name in all_model_results:
                continue

            for horizon in [7, 30, 90]:
                # Collect predictions for this model+horizon across all streams
                preds_end = []
                preds_start = []
                actuals_end = []
                actuals_start = []

                for series_name in streams:
                    key = f"{series_name}_h{horizon}"

                    if model_name == "naive":
                        # Naive has seasonal_naive and ema sub-results
                        continue
                    elif key not in model_data:
                        continue

                    # We don't have raw predictions saved — only metrics
                    # Can't recompute from Exp 1 JSON (it only saved MAE/RMSE/MASE, not raw preds)

                all_model_results.setdefault(model_name, {})[horizon] = "need_raw_predictions"

    # The Exp 1 raw JSON only saves metrics, not raw predictions
    # We need to re-run inference for the baselines
    print("\n  Exp 1 raw JSON only contains metrics, not raw predictions.")
    print("  Computing baselines directly from data...\n")

    # Compute naive baselines directly
    test_vals = test_df[streams].values.astype(float)
    train_vals = train_df[streams].values.astype(float)

    print(f"{'Model':<25} {'Horizon':>8} {'Direction':>10} {'Streams':>8}")
    print("-" * 55)

    for horizon in [7, 30, 90]:
        if len(test_vals) < horizon:
            continue

        actual_start = test_vals[0]
        actual_end = test_vals[horizon - 1]

        # EMA
        alpha = 0.1
        ema = train_vals[0].copy()
        for t in range(1, len(train_vals)):
            row = train_vals[t]
            valid = ~np.isnan(row)
            ema[valid] = alpha * row[valid] + (1 - alpha) * ema[valid]
        # EMA predicts flat line at last smoothed value
        ema_start = ema
        ema_end = ema  # Same value — EMA predicts flat
        acc, n = horizon_direction_accuracy(ema_end, actual_start, actual_end, actual_start)
        print(f"{'EMA':<25} {horizon:>5}d   {acc:>9.1%}   {n:>6}")

        # Seasonal Naive — predict last year's values
        if len(train_vals) >= 365:
            naive_start = train_vals[-365]
            naive_end = train_vals[-365 + horizon - 1]
            # Direction: does naive predict value goes up or down from where we are now?
            acc, n = horizon_direction_accuracy(naive_end, actual_start, actual_end, actual_start)
            print(f"{'Seasonal Naive':<25} {horizon:>5}d   {acc:>9.1%}   {n:>6}")

        # "Always Up" baseline (since 84% of streams go up at 90d)
        always_up_pred = actual_start + 1  # Predict slightly above start
        acc, n = horizon_direction_accuracy(always_up_pred, actual_start, actual_end, actual_start)
        print(f"{'Always Up':<25} {horizon:>5}d   {acc:>9.1%}   {n:>6}")

    print()


def eval_experiment2_checkpoints():
    """Re-evaluate Experiment 2 v2 checkpoints with corrected metric."""
    print("=" * 70)
    print("  EXPERIMENT 2 v2 — CORRECTED DIRECTIONAL ACCURACY")
    print("=" * 70)

    from src.models import MODEL_REGISTRY
    from src.revin import RevIN

    # Load raw data (same as v2 training)
    cat_df = load_categories(frozen=True)
    index_cols = sorted([c for c in cat_df.columns if 'Index' in c or 'index' in c])
    index_cols = [c for c in index_cols if '_year_ago' not in c and '_yoy' not in c]

    raw_data = pd.DataFrame(
        cat_df[index_cols].values, columns=index_cols
    ).ffill().bfill().values.astype(np.float32)

    dates = cat_df["date"].values
    test_start = np.searchsorted(dates, np.datetime64("2024-01-01"))
    context_len = 512
    num_streams = len(index_cols)

    # Architecture configs (must match training)
    configs = {
        "transformer": {"d_model": 256, "num_heads": 8, "num_layers": 6, "patch_size": 16, "dropout": 0.0},
        "s5": {"d_model": 256, "state_dim": 64, "num_layers": 6, "dropout": 0.0},
        "mamba": {"d_model": 256, "state_dim": 64, "num_layers": 6, "dropout": 0.0},
    }

    # Also need StableXLSTMForecaster
    sys.path.insert(0, str(Path(__file__).parent))
    from experiment_02_xlstm import StableXLSTMForecaster

    ckpt_dir = RESULTS_DIR / "experiment_02_v2"

    print(f"\n{'Model':<25} {'Horizon':>8} {'Direction':>10} {'Streams':>8} {'MAE':>8}")
    print("-" * 65)

    for arch_name in ["transformer", "s5", "xlstm"]:
        ckpt_path = ckpt_dir / f"{arch_name}_best.pt"
        if not ckpt_path.exists():
            print(f"{arch_name}: no checkpoint found")
            continue

        # Build model with RevIN wrapper
        if arch_name == "xlstm":
            backbone = StableXLSTMForecaster(
                num_streams=num_streams, d_model=256, hidden_dim=256,
                num_layers=4, horizon=90, dropout=0.0,
            )
        else:
            config = configs[arch_name]
            backbone = MODEL_REGISTRY[arch_name](
                num_streams=num_streams, horizon=90, **config,
            )

        # Wrap with RevIN
        class ForecastModel(torch.nn.Module):
            def __init__(self, bb, ns):
                super().__init__()
                self.backbone = bb
                self.revin = RevIN(ns)
            def forward(self, x):
                x_n = self.revin(x, "norm")
                p_n = self.backbone(x_n)
                return self.revin(p_n, "denorm")
            def count_params(self):
                return sum(p.numel() for p in self.parameters() if p.requires_grad)

        model = ForecastModel(backbone, num_streams)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Predict
        context = raw_data[test_start - context_len: test_start]
        ctx_tensor = torch.tensor(context).unsqueeze(0)

        with torch.no_grad():
            pred_90 = model(ctx_tensor).numpy()[0]  # [90, num_streams]

        pred_start = context[-1]  # Last context value = "today"

        for h in [7, 30, 90]:
            pred_end = pred_90[h - 1]
            actual_end = raw_data[test_start + h - 1]
            actual_start = raw_data[test_start - 1]  # Same "today" as pred_start

            dir_acc, n_moved = horizon_direction_accuracy(
                pred_end, pred_start, actual_end, actual_start
            )
            mae = np.mean(np.abs(pred_end - actual_end))

            label = f"Thales-{arch_name}"
            print(f"{label:<25} {h:>5}d   {dir_acc:>9.1%}   {n_moved:>6}   {mae:>7.2f}")

    print()


def main():
    eval_experiment1_baselines()
    eval_experiment2_checkpoints()

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("""
  Old metric (daily step direction): ~21% for ALL models
  New metric (horizon-level direction): see above

  The old metric was broken because CPI is flat 57.5% of days.
  The new metric asks the real question: up or down over the full horizon?
    """)


if __name__ == "__main__":
    main()
