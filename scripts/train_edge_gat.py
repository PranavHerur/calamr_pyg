#!/usr/bin/env python3
"""
Edge-Aware GAT for medical hallucination detection.
Uses edge features (flow values) to capture alignment strength.
"""

import argparse

from calamr_pyg.models.edge_aware_gat import EdgeAwareGAT
from calamr_pyg.data.datasets import load_and_split_dataset
from calamr_pyg.training import train_model


def exec(
    dataset_path: str,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.2,
    pooling: str = "attention",
    num_heads: int = 4,
    attention_activation: str = "tanh",
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 5,
    min_delta: float = 0.001,
    seed: int = 42,
    output_dir: str = "results/edge_gat",
    verbose: bool = True,
) -> dict:
    """Train and evaluate EdgeAwareGAT model."""
    train_data, val_data, test_data = load_and_split_dataset(
        dataset_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=seed,
    )

    input_dim = train_data[0].x.size(1)

    model = EdgeAwareGAT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        edge_dim=4,
        num_heads=num_heads,
        dropout=dropout,
        pooling=pooling,
        attention_activation=attention_activation,
    )

    results = train_model(
        model,
        train_data,
        val_data,
        test_data,
        model_name="EdgeAwareGAT",
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
        output_dir=output_dir,
        verbose=verbose,
    )

    results["model_config"] = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "edge_dim": 4,
        "num_heads": num_heads,
        "dropout": dropout,
        "pooling": pooling,
        "attention_activation": attention_activation,
    }
    results["dataset"]["path"] = dataset_path

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-Aware GAT")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", default="attention", choices=["attention", "mean"])
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attention_activation", default="tanh", choices=["tanh", "relu"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/edge_gat")
    args = parser.parse_args()

    exec(
        dataset_path=args.dataset_path,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
        num_heads=args.num_heads,
        attention_activation=args.attention_activation,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        output_dir=args.output_dir,
    )
