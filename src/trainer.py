"""Unified training loop for all architectures."""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    criterion = nn.MSELoss()

    for context, target in dataloader:
        context = context.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        predicted = model(context)
        loss = criterion(predicted, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Evaluate model on validation set. Returns metrics dict."""
    model.eval()
    criterion = nn.MSELoss()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    for context, target in dataloader:
        context = context.to(device)
        target = target.to(device)

        predicted = model(context)
        loss = criterion(predicted, target)
        total_loss += loss.item()
        num_batches += 1

        all_preds.append(predicted.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # MAE across all predictions
    mae = np.mean(np.abs(preds - targets))

    # Directional accuracy (per-step direction of change)
    pred_diff = np.diff(preds, axis=1)
    target_diff = np.diff(targets, axis=1)
    direction_correct = np.sign(pred_diff) == np.sign(target_diff)
    dir_acc = np.mean(direction_correct)

    return {
        "loss": total_loss / max(num_batches, 1),
        "mae": mae,
        "directional_accuracy": dir_acc,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    model_name: str = "model",
) -> dict:
    """Full training loop with early stopping.

    Returns dict with training history and best metrics.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_metrics = {}
    best_state = None
    no_improve = 0

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_dir_acc": []}

    print(f"\nTraining {model_name} ({model.count_params():,} params) on {device}")
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'Val MAE':>10} {'Val Dir%':>10} {'LR':>10}")
    print("-" * 65)

    start_time = time.time()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_dir_acc"].append(val_metrics["directional_accuracy"])

        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"{epoch+1:5d} {train_loss:12.6f} {val_metrics['loss']:12.6f} "
                f"{val_metrics['mae']:10.4f} {val_metrics['directional_accuracy']:9.2%} "
                f"{current_lr:10.2e}"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics.copy()
            best_metrics["epoch"] = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\nBest at epoch {best_metrics['epoch']}: "
          f"MAE={best_metrics['mae']:.4f}, "
          f"Dir={best_metrics['directional_accuracy']:.2%}")
    print(f"Training time: {elapsed:.1f}s")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_metrics": best_metrics,
        "training_time": elapsed,
        "model_name": model_name,
        "num_params": model.count_params(),
    }
