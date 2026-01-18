#!/usr/bin/env python3
"""
Hybrid GCN + Attention Pooling for medical hallucination detection.
Combines working GCN layers with attention-based graph pooling.
"""

import argparse
from datetime import datetime
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np

from calamr_pyg.models import HybridGCN
from calamr_pyg.data.datasets import load_and_split_dataset
from calamr_pyg.training import (
    train_epoch,
    evaluate,
    compute_metrics,
    find_optimal_threshold,
    get_youden_threshold,
    print_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Hybrid GCN + Attention Pooling")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of GCN layers"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--pooling",
        default="attention",
        choices=["attention", "mean", "max"],
        help="Pooling method",
    )
    parser.add_argument(
        "--multi_head", action="store_true", help="Use multi-head attention pooling"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay"
    )
    parser.add_argument(
        "--attention_activation",
        default="tanh",
        choices=["tanh", "relu"],
        help="Attention activation",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset directory containing .pt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split data
    train_data, val_data, test_data = load_and_split_dataset(
        args.dataset_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=args.seed
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    input_dim = train_data[0].x.size(1)

    # Model
    model = HybridGCN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
        use_multi_head_attention=args.multi_head,
        num_attention_heads=args.num_heads,
        attention_activation=args.attention_activation,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Pooling method: {args.pooling}"
        + (f" (multi-head, {args.num_heads} heads)" if args.multi_head else "")
    )

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Make output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Training
    best_val_f1 = 0
    patience_counter = 0

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        _, train_f1 = find_optimal_threshold(train_labels, train_preds)
        val_threshold, val_f1 = find_optimal_threshold(val_labels, val_preds)

        scheduler.step(val_f1)

        print(
            f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"Train F1={train_f1:.4f}, Val F1={val_f1:.4f}, LR={optimizer.param_groups[0]['lr']:.6f},\n"
            f"\tTime={time.time() - epoch_start_time:.2f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), results_dir / "best_hybrid_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Load best model
    model.load_state_dict(torch.load(results_dir / "best_hybrid_model.pt"))

    # Evaluate on test set
    test_loss, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    # Optimal threshold from val and compute metrics on test set
    val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    val_threshold, _ = find_optimal_threshold(val_labels, val_preds)
    threshold_from_val_metrics = compute_metrics(
        test_labels, test_preds, val_threshold
    )
    print_metrics(threshold_from_val_metrics, "Val Threshold Metrics")

    # Optimal threshold from val using Youden's method
    youden_val_threshold = get_youden_threshold(val_labels, val_preds)
    youden_val_metrics = compute_metrics(test_labels, test_preds, youden_val_threshold)
    print_metrics(youden_val_metrics, "Youden's Val Metrics")

    # Optimal threshold from test set
    test_threshold, _ = find_optimal_threshold(test_labels, test_preds)
    threshold_from_test_metrics = compute_metrics(
        test_labels, test_preds, test_threshold
    )
    print_metrics(threshold_from_test_metrics, "Test Threshold Metrics")

    # Save results
    results = {
        "model_config": {
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "pooling": args.pooling,
            "multi_head": args.multi_head,
            "num_heads": args.num_heads,
            "attention_activation": args.attention_activation,
        },
        "training_config": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
        },
        "dataset": {
            "path": args.dataset_path,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
        },
        "results": {
            "best_val_f1": best_val_f1,
            "threshold_from_test_metrics": threshold_from_test_metrics,
            "threshold_from_val_metrics": threshold_from_val_metrics,
            "youden_val_metrics": youden_val_metrics,
            "training_time": training_time,
        },
    }

    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir}")
    print(f"Best model saved to {results_dir / 'best_hybrid_model.pt'}")


if __name__ == "__main__":
    main()
