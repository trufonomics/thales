"""Experiment 1 — Baseline Zoo.

Runs TimesFM, Chronos, and Moirai zero-shot on Truflation CPI streams.
Evaluates at 7, 30, 90, 365-day horizons.
Also runs naive baselines (seasonal naive, EMA) for comparison.

Usage:
    python scripts/experiment_01_baselines.py --models timesfm chronos moirai --horizons 7 30 90

Run on Vast.ai with A100. Estimated time: 30-60 minutes for all models.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    load_categories,
    extract_index_series,
    train_test_split_temporal,
    prepare_series_for_model,
    TRUFLATION_CATEGORIES,
)
from src.metrics import evaluate_forecast


RESULTS_DIR = Path(__file__).parent.parent / "results" / "experiment_01"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [7, 30, 90]
TEST_START = "2024-01-01"


def seasonal_naive(train: np.ndarray, horizon: int, period: int = 365) -> np.ndarray:
    """Predict by repeating the value from one year ago."""
    if len(train) < period:
        return np.full(horizon, train[-1])
    return np.array([train[-(period - (i % period))] for i in range(horizon)])


def ema_forecast(train: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    """Exponential moving average forecast (flat line from last EMA value)."""
    ema = train[0]
    for v in train[1:]:
        if not np.isnan(v):
            ema = alpha * v + (1 - alpha) * ema
    return np.full(horizon, ema)


def run_naive_baselines(series_list, train_df, test_df, horizons):
    """Run seasonal naive and EMA baselines."""
    results = {}

    for series in tqdm(series_list, desc="Naive baselines"):
        name = series["name"]
        train_vals = train_df[name].ffill().bfill().values.astype(float)
        test_vals = test_df[name].ffill().bfill().values.astype(float)

        for h in horizons:
            if len(test_vals) < h:
                continue

            actual = test_vals[:h]

            naive_pred = seasonal_naive(train_vals, h)
            ema_pred = ema_forecast(train_vals, h)

            key = f"{name}_h{h}"
            results[key] = {
                "series": name,
                "horizon": h,
                "seasonal_naive": evaluate_forecast(actual, naive_pred, train_vals),
                "ema": evaluate_forecast(actual, ema_pred, train_vals),
            }

    return results


def run_chronos(series_list, train_df, test_df, horizons, model_id="amazon/chronos-bolt-small"):
    """Run Amazon Chronos zero-shot.

    Args:
        model_id: HuggingFace model ID. Options:
            - amazon/chronos-bolt-small (48M, recommended — faster, direct quantile)
            - amazon/chronos-bolt-base (205M)
            - amazon/chronos-t5-small (46M, original tokenization approach)
    """
    try:
        from chronos import ChronosPipeline, ChronosBoltPipeline
    except ImportError:
        print("chronos-forecasting not installed. pip install chronos-forecasting")
        return {}

    model_name = model_id.split("/")[-1]
    print(f"Loading {model_name}...")
    is_bolt = "bolt" in model_id.lower()
    pipeline_cls = ChronosBoltPipeline if is_bolt else ChronosPipeline
    pipeline = pipeline_cls.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float32,
    )

    results = {}
    for series in tqdm(series_list, desc=model_name):
        name = series["name"]
        train_vals = train_df[name].ffill().bfill().values.astype(float)
        test_vals = test_df[name].ffill().bfill().values.astype(float)
        context = torch.tensor(train_vals, dtype=torch.float32)

        for h in horizons:
            if len(test_vals) < h:
                continue

            if is_bolt:
                forecast = pipeline.predict(context, prediction_length=h)
                median_pred = forecast.numpy().flatten()[:h]
            else:
                forecast = pipeline.predict(context, h, num_samples=20)
                median_pred = np.median(forecast.numpy(), axis=1).flatten()[:h]
            actual = test_vals[:h]

            key = f"{name}_h{h}"
            results[key] = {
                "series": name,
                "horizon": h,
                "metrics": evaluate_forecast(actual, median_pred, train_vals),
            }

    return results


def run_timesfm(series_list, train_df, test_df, horizons):
    """Run Google TimesFM zero-shot."""
    try:
        import timesfm
    except ImportError:
        print("timesfm not installed. pip install timesfm")
        return {}

    print("Loading TimesFM...")
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu" if torch.cuda.is_available() else "cpu",
            per_core_batch_size=32,
            horizon_len=max(horizons),
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-200m-pytorch",
        ),
    )

    results = {}
    for series in tqdm(series_list, desc="TimesFM"):
        name = series["name"]
        train_vals = train_df[name].ffill().bfill().values.astype(float)
        test_vals = test_df[name].ffill().bfill().values.astype(float)

        point_forecast, _ = tfm.forecast(
            [train_vals],
            freq=[0],
        )

        for h in horizons:
            if len(test_vals) < h:
                continue

            predicted = point_forecast[0][:h]
            actual = test_vals[:h]

            key = f"{name}_h{h}"
            results[key] = {
                "series": name,
                "horizon": h,
                "metrics": evaluate_forecast(actual, predicted, train_vals),
            }

    return results


def run_moirai(series_list, train_df, test_df, horizons, model_size="small"):
    """Run Salesforce Moirai zero-shot."""
    try:
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    except ImportError:
        print("uni2ts not installed. pip install uni2ts")
        return {}

    print(f"Loading Moirai-{model_size}...")
    model = MoiraiModule.from_pretrained(
        f"Salesforce/moirai-1.1-R-{model_size}"
    )

    results = {}
    for series in tqdm(series_list, desc=f"Moirai-{model_size}"):
        name = series["name"]
        train_vals = train_df[name].ffill().bfill().values.astype(float)
        test_vals = test_df[name].ffill().bfill().values.astype(float)

        for h in horizons:
            if len(test_vals) < h:
                continue

            try:
                forecast_module = MoiraiForecast(
                    module=model,
                    prediction_length=h,
                    context_length=min(len(train_vals), 2048),
                    patch_size="auto",
                    num_samples=20,
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                )

                from gluonts.dataset.common import ListDataset
                predictor = forecast_module.create_predictor(batch_size=1)
                ds = ListDataset(
                    [{
                        "start": pd.Timestamp(train_df["date"].iloc[0]),
                        "target": train_vals,
                    }],
                    freq="D",
                )

                forecasts = list(predictor.predict(ds))
                median_pred = np.median(
                    forecasts[0].samples, axis=0
                ).flatten()[:h]
                actual = test_vals[:h]

                key = f"{name}_h{h}"
                results[key] = {
                    "series": name,
                    "horizon": h,
                    "metrics": evaluate_forecast(actual, median_pred, train_vals),
                }
            except Exception as e:
                print(f"Moirai failed on {name} h={h}: {e}")
                continue

    return results


def run_tirex(series_list, train_df, test_df, horizons):
    """Run NX-AI TiRex (xLSTM-based, 35M params) zero-shot.

    Requires: pip install git+https://github.com/NX-AI/tirex.git
    Also requires HuggingFace authentication (model is gated).
    """
    try:
        from tirex import TiRexZero
    except ImportError:
        print("tirex not installed. pip install git+https://github.com/NX-AI/tirex.git")
        return {}

    print("Loading TiRex...")
    try:
        model = TiRexZero.from_pretrained("NX-AI/TiRex-xl-1.1", backend="torch")
    except Exception as e:
        print(f"TiRex load failed (may need HuggingFace auth): {e}")
        return {}

    results = {}
    for series in tqdm(series_list, desc="TiRex"):
        name = series["name"]
        train_vals = train_df[name].ffill().bfill().values.astype(float)
        test_vals = test_df[name].ffill().bfill().values.astype(float)

        for h in horizons:
            if len(test_vals) < h:
                continue
            try:
                context_len = min(len(train_vals), 2048)
                context = torch.tensor(train_vals[-context_len:], dtype=torch.float32).unsqueeze(0)
                out = model.forecast(context, prediction_length=h)
                median_pred = out.median(dim=-1).values.numpy().flatten()[:h]
                actual = test_vals[:h]

                key = f"{name}_h{h}"
                results[key] = {
                    "series": name,
                    "horizon": h,
                    "metrics": evaluate_forecast(actual, median_pred, train_vals),
                }
            except Exception as e:
                print(f"TiRex failed on {name} h={h}: {e}")
                continue

    return results


def run_toto(series_list, train_df, test_df, horizons):
    """Run Datadog Toto (151M params) zero-shot."""
    try:
        from toto.model.toto import Toto
        from toto.inference.forecaster import TotoForecaster
        from toto.data.util.dataset import MaskedTimeseries
    except ImportError:
        print("toto not installed. pip install git+https://github.com/DataDog/toto.git")
        return {}

    print("Loading Toto...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        toto_model = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
        toto_model = toto_model.to(device).eval()
        forecaster = TotoForecaster(model=toto_model.backbone)
    except Exception as e:
        print(f"Toto load failed: {e}")
        return {}

    DAY_SECONDS = 86400

    results = {}
    for series in tqdm(series_list, desc="Toto"):
        name = series["name"]
        train_vals = train_df[name].ffill().bfill().values.astype(float)
        test_vals = test_df[name].ffill().bfill().values.astype(float)

        context_len = min(len(train_vals), 4096)
        vals = train_vals[-context_len:]

        start_ts = int(pd.Timestamp(train_df["date"].iloc[-context_len]).timestamp())
        timestamps = np.array([start_ts + i * DAY_SECONDS for i in range(context_len)])

        series_tensor = torch.tensor(vals, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        padding_mask = torch.ones(1, 1, context_len, dtype=torch.bool).to(device)
        id_mask = torch.zeros(1, 1, context_len, dtype=torch.int64).to(device)
        ts_tensor = torch.tensor(timestamps, dtype=torch.int64).unsqueeze(0).unsqueeze(0).to(device)
        interval = torch.tensor([[DAY_SECONDS]], dtype=torch.int64).to(device)

        inputs = MaskedTimeseries(
            series=series_tensor,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=ts_tensor,
            time_interval_seconds=interval,
            num_exogenous_variables=0,
        )

        for h in horizons:
            if len(test_vals) < h:
                continue
            try:
                forecast = forecaster.forecast(inputs, prediction_length=h, num_samples=20)
                samples = forecast.samples.cpu().numpy()
                median_pred = np.median(samples, axis=0).flatten()[:h]
                actual = test_vals[:h]

                key = f"{name}_h{h}"
                results[key] = {
                    "series": name,
                    "horizon": h,
                    "metrics": evaluate_forecast(actual, median_pred, train_vals),
                }
            except Exception as e:
                print(f"Toto failed on {name} h={h}: {e}")
                continue

    return results


def aggregate_results(all_results: dict) -> pd.DataFrame:
    """Aggregate per-series results into summary table."""
    rows = []
    for model_name, model_results in all_results.items():
        for key, result in model_results.items():
            metrics = result.get("metrics", result.get("seasonal_naive", {}))
            if "seasonal_naive" in result:
                for baseline_name in ["seasonal_naive", "ema"]:
                    m = result[baseline_name]
                    rows.append({
                        "model": baseline_name,
                        "series": result["series"],
                        "horizon": result["horizon"],
                        **m,
                    })
            else:
                rows.append({
                    "model": model_name,
                    "series": result["series"],
                    "horizon": result["horizon"],
                    **metrics,
                })

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Baseline Zoo")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["naive"],
        choices=["naive", "chronos-bolt", "chronos-t5", "timesfm", "moirai", "tirex", "toto", "all"],
        help="Which models to run",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=HORIZONS,
        help="Forecast horizons in days",
    )
    parser.add_argument(
        "--frozen",
        action="store_true",
        default=True,
        help="Use frozen (point-in-time) data for training",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default=TEST_START,
        help="Test set start date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    if "all" in args.models:
        args.models = ["naive", "chronos-bolt", "chronos-t5", "moirai", "tirex", "toto"]

    print(f"Loading Truflation data (frozen={args.frozen})...")
    cat_df = load_categories(frozen=args.frozen)
    index_df = extract_index_series(cat_df)

    print(f"Splitting at {args.test_start}...")
    train_df, test_df = train_test_split_temporal(index_df, args.test_start)
    print(f"Train: {len(train_df)} days ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Test:  {len(test_df)} days ({test_df['date'].min()} to {test_df['date'].max()})")

    series_list = prepare_series_for_model(train_df)
    print(f"Prepared {len(series_list)} series for evaluation")
    print(f"Horizons: {args.horizons}")

    all_results = {}

    if "naive" in args.models:
        print("\n=== Running naive baselines ===")
        all_results["naive"] = run_naive_baselines(
            series_list, train_df, test_df, args.horizons
        )

    if "chronos-bolt" in args.models:
        print("\n=== Running Chronos-Bolt (48M, direct quantile) ===")
        all_results["chronos-bolt"] = run_chronos(
            series_list, train_df, test_df, args.horizons,
            model_id="amazon/chronos-bolt-small",
        )

    if "chronos-t5" in args.models:
        print("\n=== Running Chronos-T5 (46M, tokenized) ===")
        all_results["chronos-t5"] = run_chronos(
            series_list, train_df, test_df, args.horizons,
            model_id="amazon/chronos-t5-small",
        )

    if "timesfm" in args.models:
        print("\n=== Running TimesFM ===")
        all_results["timesfm"] = run_timesfm(
            series_list, train_df, test_df, args.horizons
        )

    if "moirai" in args.models:
        print("\n=== Running Moirai (14M) ===")
        all_results["moirai"] = run_moirai(
            series_list, train_df, test_df, args.horizons
        )

    if "tirex" in args.models:
        print("\n=== Running TiRex (35M, xLSTM) ===")
        all_results["tirex"] = run_tirex(
            series_list, train_df, test_df, args.horizons
        )

    if "toto" in args.models:
        print("\n=== Running Toto (151M, Datadog) ===")
        all_results["toto"] = run_toto(
            series_list, train_df, test_df, args.horizons
        )

    print("\n=== Aggregating results ===")
    summary = aggregate_results(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / f"baseline_results_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    raw_path = RESULTS_DIR / f"baseline_raw_{timestamp}.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {raw_path}")

    print("\n=== Summary (mean across series) ===")
    if not summary.empty:
        agg = summary.groupby(["model", "horizon"])[["mae", "rmse", "mase", "directional_accuracy"]].mean()
        print(agg.round(4).to_string())


if __name__ == "__main__":
    main()
