"""Re-run TSFM baselines and compute corrected horizon-level direction.

Runs each model, saves raw predictions, computes:
- MAE (same as before)
- Horizon direction: is predicted end > or < start? Matches actual?

Usage:
    pip install chronos-forecasting
    python scripts/eval_baselines_direction.py --models chronos-t5 chronos-bolt
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_categories, extract_index_series, train_test_split_temporal

RESULTS_DIR = Path(__file__).parent.parent / "results" / "experiment_01_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HORIZONS = [7, 30, 90]
TEST_START = "2024-01-01"


def horizon_direction(pred_series, actual_series, start_value):
    """Compute horizon-level direction accuracy.

    pred_series: [horizon] predicted values
    actual_series: [horizon] actual values
    start_value: the value at the start (today)

    Returns: 1 if predicted direction matches actual, 0 if not, -1 if actual flat
    """
    pred_dir = np.sign(pred_series[-1] - start_value)
    actual_dir = np.sign(actual_series[-1] - start_value)
    if actual_dir == 0:
        return -1  # Flat — exclude from scoring
    return int(pred_dir == actual_dir)


def run_model(model_name, model_id, series_list, train_df, test_df, horizons):
    """Run a single model and return predictions + metrics."""

    if model_name.startswith("chronos"):
        from chronos import ChronosPipeline, ChronosBoltPipeline
        is_bolt = "bolt" in model_id
        cls = ChronosBoltPipeline if is_bolt else ChronosPipeline
        pipeline = cls.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float32)

        results = {}
        for series in tqdm(series_list, desc=model_name):
            name = series["name"]
            train_vals = train_df[name].ffill().bfill().values.astype(float)
            test_vals = test_df[name].ffill().bfill().values.astype(float)
            context = torch.tensor(train_vals, dtype=torch.float32)
            start_val = train_vals[-1]

            for h in horizons:
                if len(test_vals) < h:
                    continue

                if is_bolt:
                    forecast = pipeline.predict(context, prediction_length=h)
                    pred = forecast.numpy().flatten()[:h]
                else:
                    forecast = pipeline.predict(context, h, num_samples=20)
                    pred = np.median(forecast.numpy(), axis=1).flatten()[:h]

                actual = test_vals[:h]
                mae = np.mean(np.abs(pred - actual))
                h_dir = horizon_direction(pred, actual, start_val)

                results[f"{name}_h{h}"] = {
                    "series": name, "horizon": h,
                    "mae": float(mae),
                    "horizon_direction": h_dir,
                    "pred_end": float(pred[-1]),
                    "actual_end": float(actual[-1]),
                    "start": float(start_val),
                }

        return results

    elif model_name == "moirai":
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        from gluonts.dataset.common import ListDataset
        model = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")

        results = {}
        for series in tqdm(series_list, desc="moirai"):
            name = series["name"]
            train_vals = train_df[name].ffill().bfill().values.astype(float)
            test_vals = test_df[name].ffill().bfill().values.astype(float)
            start_val = train_vals[-1]

            for h in horizons:
                if len(test_vals) < h:
                    continue
                try:
                    fm = MoiraiForecast(
                        module=model, prediction_length=h,
                        context_length=min(len(train_vals), 2048),
                        patch_size="auto", num_samples=20,
                        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
                    )
                    predictor = fm.create_predictor(batch_size=1)
                    ds = ListDataset([{
                        "start": pd.Timestamp(train_df["date"].iloc[0]),
                        "target": train_vals,
                    }], freq="D")
                    forecasts = list(predictor.predict(ds))
                    pred = np.median(forecasts[0].samples, axis=0).flatten()[:h]
                    actual = test_vals[:h]
                    mae = np.mean(np.abs(pred - actual))
                    h_dir = horizon_direction(pred, actual, start_val)

                    results[f"{name}_h{h}"] = {
                        "series": name, "horizon": h,
                        "mae": float(mae), "horizon_direction": h_dir,
                        "pred_end": float(pred[-1]), "actual_end": float(actual[-1]),
                        "start": float(start_val),
                    }
                except Exception as e:
                    print(f"  Moirai failed {name} h={h}: {e}")
        return results

    elif model_name == "tirex":
        from tirex import TiRexZero
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        model = TiRexZero.from_pretrained("NX-AI/TiRex", backend="torch")

        results = {}
        for series in tqdm(series_list, desc="tirex"):
            name = series["name"]
            train_vals = train_df[name].ffill().bfill().values.astype(float)
            test_vals = test_df[name].ffill().bfill().values.astype(float)
            start_val = train_vals[-1]
            context = train_vals[-2048:].astype(np.float32)

            for h in horizons:
                if len(test_vals) < h:
                    continue
                try:
                    out = model.forecast(context, prediction_length=h)
                    quantiles, median = out
                    pred = median.numpy().flatten()[:h]
                    actual = test_vals[:h]
                    mae = np.mean(np.abs(pred - actual))
                    h_dir = horizon_direction(pred, actual, start_val)

                    results[f"{name}_h{h}"] = {
                        "series": name, "horizon": h,
                        "mae": float(mae), "horizon_direction": h_dir,
                        "pred_end": float(pred[-1]), "actual_end": float(actual[-1]),
                        "start": float(start_val),
                    }
                except Exception as e:
                    print(f"  TiRex failed {name} h={h}: {e}")
        return results

    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["chronos-t5"],
                       choices=["chronos-t5", "chronos-bolt", "moirai", "tirex", "all"])
    args = parser.parse_args()

    if "all" in args.models:
        args.models = ["chronos-t5", "chronos-bolt", "moirai", "tirex"]

    MODEL_IDS = {
        "chronos-t5": "amazon/chronos-t5-small",
        "chronos-bolt": "amazon/chronos-bolt-small",
    }

    print("Loading data...")
    cat_df = load_categories(frozen=True)
    idx_df = extract_index_series(cat_df)
    train_df, test_df = train_test_split_temporal(idx_df, TEST_START)
    series_list = [{"name": c} for c in idx_df.columns if c != "date"]
    print(f"Streams: {len(series_list)}, Train: {len(train_df)}d, Test: {len(test_df)}d")

    all_results = {}

    for model_name in args.models:
        print(f"\n=== {model_name.upper()} ===")
        model_id = MODEL_IDS.get(model_name, model_name)
        results = run_model(model_name, model_id, series_list, train_df, test_df, HORIZONS)
        all_results[model_name] = results

    # Summary
    print(f"\n{'='*70}")
    print(f"  CORRECTED DIRECTIONAL ACCURACY — TSFM BASELINES")
    print(f"{'='*70}")
    print(f"\n{'Model':<20} {'Horizon':>8} {'MAE':>8} {'Direction':>10} {'Scored':>8}")
    print("-" * 58)

    for model_name, results in all_results.items():
        for h in HORIZONS:
            h_results = [v for v in results.values() if v["horizon"] == h]
            if not h_results:
                continue
            maes = [r["mae"] for r in h_results]
            dirs = [r["horizon_direction"] for r in h_results if r["horizon_direction"] != -1]
            dir_acc = np.mean(dirs) if dirs else 0
            print(f"{model_name:<20} {h:>5}d   {np.mean(maes):>7.2f}   {dir_acc:>9.1%}   {len(dirs):>6}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"direction_results_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
