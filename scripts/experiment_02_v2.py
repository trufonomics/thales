"""Experiment 2 v2 — Architecture Selection with Fixed Training Protocol.

Fixes from v1:
1. RevIN (per-window normalization) instead of global z-score
2. Composite loss (Huber + Trend + Direction) instead of MSE
3. Loss computed on raw-scale values after denormalization
4. Rolling evaluation matching Experiment 1 protocol

Usage:
    python scripts/experiment_02_v2.py --arch all --horizon 90
    python scripts/experiment_02_v2.py --arch transformer --horizon 30
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_categories, extract_index_series, train_test_split_temporal
from src.revin import RevIN
from src.losses import CompositeLoss
from src.metrics import evaluate_forecast


RESULTS_DIR = Path(__file__).parent.parent / "results" / "experiment_02_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Dataset ───────────────────────────────────────────────────────────

class RawTimeSeriesDataset(Dataset):
    """Sliding windows on RAW (unnormalized) data.

    RevIN handles normalization per-window inside the model.
    """

    def __init__(self, data: np.ndarray, context_len: int, horizon: int, stride: int = 1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.context_len = context_len
        self.horizon = horizon
        self.total_len = context_len + horizon
        self.stride = stride
        self.num_windows = max(0, (len(data) - self.total_len) // stride + 1)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        context = self.data[start : start + self.context_len]
        target = self.data[start + self.context_len : start + self.total_len]
        return context, target


# ─── Model Wrapper with RevIN ──────────────────────────────────────────

class ForecastModel(nn.Module):
    """Wraps any backbone with RevIN and exposes a clean interface."""

    def __init__(self, backbone: nn.Module, num_streams: int):
        super().__init__()
        self.backbone = backbone
        self.revin = RevIN(num_streams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, context_len, num_streams] — RAW values
        x_norm = self.revin(x, mode="norm")
        pred_norm = self.backbone(x_norm)
        pred_raw = self.revin(pred_norm, mode="denorm")
        return pred_raw

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Architectures ─────────────────────────────────────────────────────

def build_model(arch_name: str, num_streams: int, horizon: int) -> ForecastModel:
    """Build a model with RevIN wrapper."""
    from src.models import MODEL_REGISTRY

    configs = {
        "transformer": {"d_model": 256, "num_heads": 8, "num_layers": 6, "patch_size": 16, "dropout": 0.1},
        "s5":          {"d_model": 256, "state_dim": 64, "num_layers": 6, "dropout": 0.1},
        "mamba":       {"d_model": 256, "state_dim": 64, "num_layers": 6, "dropout": 0.1},
    }

    if arch_name == "xlstm":
        from experiment_02_xlstm import StableXLSTMForecaster
        backbone = StableXLSTMForecaster(
            num_streams=num_streams, d_model=256, hidden_dim=256,
            num_layers=4, horizon=horizon, dropout=0.1,
        )
    else:
        config = configs[arch_name]
        backbone = MODEL_REGISTRY[arch_name](
            num_streams=num_streams, horizon=horizon, **config,
        )

    return ForecastModel(backbone, num_streams)


# ─── Training ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    n = 0
    for context, target in loader:
        context, target = context.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(context)
        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    n = 0
    for context, target in loader:
        context, target = context.to(device), target.to(device)
        pred = model(context)
        total_loss += criterion(pred, target).item()
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        n += 1

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    mae = np.mean(np.abs(preds - targets))
    pred_diff = np.diff(preds, axis=1)
    target_diff = np.diff(targets, axis=1)
    dir_acc = np.mean(np.sign(pred_diff) == np.sign(target_diff))

    return {
        "loss": total_loss / max(n, 1),
        "mae": mae,
        "directional_accuracy": dir_acc,
    }


def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3, patience=15, name="model"):
    model = model.to(device)
    criterion = CompositeLoss(lambda_trend=0.5, lambda_direction=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_metrics = {}
    best_state = None
    no_improve = 0

    print(f"\nTraining {name} ({model.count_params():,} params) on {device}")
    print(f"Loss: Huber + 0.5*TrendConsistency + 0.3*Directional")
    print(f"{'Epoch':>5} {'Train':>10} {'Val':>10} {'MAE':>10} {'Dir%':>8} {'LR':>10}")
    print("-" * 58)

    t0 = time.time()
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val = validate(model, val_loader, criterion, device)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"{epoch+1:5d} {train_loss:10.4f} {val['loss']:10.4f} "
                  f"{val['mae']:10.2f} {val['directional_accuracy']:7.2%} {lr_now:10.2e}")

        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_metrics = {**val, "epoch": epoch + 1}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    print(f"\nBest epoch {best_metrics.get('epoch', '?')}: "
          f"MAE={best_metrics.get('mae', 0):.2f}, "
          f"Dir={best_metrics.get('directional_accuracy', 0):.2%}")
    print(f"Time: {elapsed:.0f}s")

    if best_state:
        model.load_state_dict(best_state)

    return {"best_metrics": best_metrics, "time": elapsed}


# ─── Test Evaluation (matches Experiment 1 protocol) ───────────────────

@torch.no_grad()
def test_eval(model, raw_data, test_start, context_len, horizons, device):
    """Evaluate on test set in RAW space, matching Exp 1 metrics."""
    model.eval()
    results = {}

    for h in horizons:
        context = raw_data[test_start - context_len : test_start]
        actual = raw_data[test_start : test_start + h]
        if len(actual) < h:
            continue

        ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(ctx).cpu().numpy()[0][:h]

        num_streams = pred.shape[1]
        train_raw = raw_data[:test_start]
        maes, mases, dirs = [], [], []

        for s in range(num_streams):
            sp, sa, st = pred[:, s], actual[:, s], train_raw[:, s]
            mae = np.mean(np.abs(sp - sa))
            naive_err = np.abs(st[365:] - st[:-365])
            naive_mae = np.mean(naive_err) if len(naive_err) > 0 else 1.0
            mase = mae / naive_mae if naive_mae > 0 else float("inf")
            dir_acc = np.mean(np.sign(np.diff(sp)) == np.sign(np.diff(sa))) if h > 1 else 0.5

            maes.append(mae)
            mases.append(mase)
            dirs.append(dir_acc)

        results[h] = {
            "mae": float(np.mean(maes)),
            "mase": float(np.mean(mases)),
            "directional_accuracy": float(np.mean(dirs)),
        }
        print(f"  {h:3d}d — MAE: {results[h]['mae']:.4f}  "
              f"MASE: {results[h]['mase']:.4f}  "
              f"Dir: {results[h]['directional_accuracy']:.2%}")

    return results


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", nargs="+", default=["all"],
                       choices=["transformer", "s5", "mamba", "xlstm", "all"])
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

    # Load RAW data (no normalization — RevIN handles it per-window)
    print("Loading Truflation data (RAW, no global normalization)...")
    cat_df = load_categories(frozen=True)
    index_cols = sorted([c for c in cat_df.columns if 'Index' in c or 'index' in c])
    index_cols = [c for c in index_cols if '_year_ago' not in c and '_yoy' not in c]

    raw_data = pd.DataFrame(
        cat_df[index_cols].values, columns=index_cols
    ).ffill().bfill().values.astype(np.float32)

    dates = cat_df["date"].values
    train_end = np.searchsorted(dates, np.datetime64("2023-01-01"))
    val_end = np.searchsorted(dates, np.datetime64("2024-01-01"))

    num_streams = len(index_cols)
    print(f"Streams: {num_streams}, Train: {train_end}d, Val: {val_end-train_end}d, Test: {len(raw_data)-val_end}d")

    # Datasets on RAW values
    train_ds = RawTimeSeriesDataset(raw_data[:train_end], args.context_len, args.horizon, stride=7)
    val_ds = RawTimeSeriesDataset(raw_data[train_end - args.context_len : val_end], args.context_len, args.horizon, stride=7)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    all_results = {}

    for arch in args.arch:
        print(f"\n{'='*60}")
        print(f"  {arch.upper()} + RevIN + CompositeLoss")
        print(f"{'='*60}")

        lr = 3e-4 if arch == "xlstm" else args.lr
        model = build_model(arch, num_streams, args.horizon)

        result = train_model(
            model, train_loader, val_loader, args.device,
            epochs=args.epochs, lr=lr, patience=args.patience, name=arch,
        )

        print(f"\n  Test evaluation:")
        test = test_eval(model, raw_data, val_end, args.context_len, [7, 30, 90], args.device)

        all_results[arch] = {
            "params": model.count_params(),
            "train_result": result,
            "test": test,
        }

        torch.save(model.state_dict(), RESULTS_DIR / f"{arch}_best.pt")

    # Summary
    print(f"\n{'='*80}")
    print("  EXPERIMENT 2 v2 — RESULTS (RevIN + Composite Loss)")
    print(f"{'='*80}")

    baselines = {
        "TiRex (35M)": {7: {"mae": 1.07}, 30: {"mae": 1.41}, 90: {"mae": 1.72, "dir": 0.21}},
        "EMA": {7: {"mae": 1.09}, 30: {"mae": 1.37}, 90: {"mae": 1.90, "dir": 0.60}},
        "Chronos-T5 (46M)": {7: {"mae": 1.05}, 30: {"mae": 1.52}, 90: {"mae": 2.06, "dir": 0.58}},
        "Seasonal Naive": {7: {"mae": 4.13}, 30: {"mae": 4.34}, 90: {"mae": 3.98, "dir": 0.71}},
    }

    print(f"\n{'Model':<30} {'7d MAE':>8} {'30d MAE':>8} {'90d MAE':>8} {'90d MASE':>9} {'90d Dir':>8}")
    print("-" * 75)
    for name, res in all_results.items():
        t = res["test"]
        label = f"Thales-{name} ({res['params']/1e6:.1f}M)"
        print(f"{label:<30} {t.get(7,{}).get('mae',0):>8.2f} {t.get(30,{}).get('mae',0):>8.2f} "
              f"{t.get(90,{}).get('mae',0):>8.2f} {t.get(90,{}).get('mase',0):>9.4f} "
              f"{t.get(90,{}).get('directional_accuracy',0):>7.2%}")
    print("-" * 75)
    for name, b in baselines.items():
        print(f"{name:<30} {b[7]['mae']:>8.2f} {b[30]['mae']:>8.2f} {b[90]['mae']:>8.2f} "
              f"{'':>9} {b[90].get('dir',0):>7.2%}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"results_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
