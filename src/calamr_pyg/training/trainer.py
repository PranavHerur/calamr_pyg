"""Training and evaluation functions."""

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Optional, Sequence, Any

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np

from calamr_pyg.training.metrics import (
    compute_metrics,
    find_optimal_threshold,
    get_youden_threshold,
    print_metrics,
)


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Tuple of (average_loss, predictions, labels)
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    debug_first_batch = True

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss = criterion(out, batch.y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = torch.sigmoid(out)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

            if debug_first_batch:
                print(
                    f"  Raw outputs: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}"
                )
                print(
                    f"  Predictions: min={preds.min().item():.4f}, max={preds.max().item():.4f}, mean={preds.mean().item():.4f}"
                )
                debug_first_batch = False

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on validation/test data.

    Args:
        model: PyTorch model
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average_loss, predictions, labels)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.float())
            total_loss += loss.item()

            preds = torch.sigmoid(out)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)


def train_model(
    model: nn.Module,
    train_data,
    val_data,
    test_data,
    *,
    model_name: str = "Model",
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 5,
    min_delta: float = 0.001,
    seed: int = 42,
    output_dir: str = "results",
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Unified training function for all GNN models.

    Args:
        model: PyTorch model to train
        train_data: Training dataset (list of Data objects)
        val_data: Validation dataset
        test_data: Test dataset
        model_name: Name of the model for logging
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        epochs: Maximum number of training epochs
        batch_size: Batch size
        patience: Early stopping patience (epochs without improvement)
        min_delta: Minimum improvement in val_loss to reset patience
        seed: Random seed
        output_dir: Directory to save results
        verbose: Print progress during training
        device: Device to train on (default: auto-detect)

    Returns:
        Dictionary containing training config and results
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if verbose:
        print(f"Using device: {device}")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    if verbose:
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model: {model_name}")
        print(f"Parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Training loop with early stopping on val_loss
    best_val_loss = float("inf")
    patience_counter = 0

    if verbose:
        print("\nStarting training...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        _, train_f1 = find_optimal_threshold(train_labels, train_preds)
        _, val_f1 = find_optimal_threshold(val_labels, val_preds)

        scheduler.step(val_loss)

        if verbose:
            print(
                f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Train F1={train_f1:.4f}, Val F1={val_f1:.4f}, "
                f"Time={time.time() - epoch_start:.2f}s"
            )

        # Early stopping on val_loss
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), results_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time
    if verbose:
        print(f"Training completed in {training_time:.2f} seconds")

    # Load best model and evaluate
    model.load_state_dict(torch.load(results_dir / "best_model.pt", weights_only=True))
    _, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    _, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

    # Compute metrics with different threshold strategies
    val_threshold, _ = find_optimal_threshold(val_labels, val_preds)
    threshold_from_val_metrics = compute_metrics(test_labels, test_preds, val_threshold)
    if verbose:
        print_metrics(threshold_from_val_metrics, "Test Metrics (Val Threshold)")

    youden_threshold = get_youden_threshold(val_labels, val_preds)
    youden_val_metrics = compute_metrics(test_labels, test_preds, youden_threshold)
    if verbose:
        print_metrics(youden_val_metrics, "Test Metrics (Youden)")

    test_threshold, _ = find_optimal_threshold(test_labels, test_preds)
    threshold_from_test_metrics = compute_metrics(test_labels, test_preds, test_threshold)

    # Build results dictionary
    results = {
        "model": model_name,
        "training_config": {
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "min_delta": min_delta,
            "seed": seed,
        },
        "dataset": {
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
        },
        "results": {
            "best_val_loss": best_val_loss,
            "threshold_from_test_metrics": threshold_from_test_metrics,
            "threshold_from_val_metrics": threshold_from_val_metrics,
            "youden_val_metrics": youden_val_metrics,
            "training_time": training_time,
            "num_params": num_params,
        },
        "output_dir": str(results_dir),
    }

    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nResults saved to {results_dir}")

    return results
