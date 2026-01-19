#!/usr/bin/env python3
"""
Run all GNN model experiments sequentially and generate analytics.
Compares performance across models: HybridGCN, EdgeAwareGAT, GraphSAGE, GIN, GatedGNN, GraphTransformer, GPS.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Optional

# Import training functions from all scripts
from train_hybrid import exec as train_hybrid
from train_edge_gat import exec as train_edge_gat
from train_sage import exec as train_sage
from train_gin import exec as train_gin
from train_gated import exec as train_gated
from train_graph_transformer import exec as train_graph_transformer
from train_gps import exec as train_gps
from train_pna import exec as train_pna


# Model registry with default configurations
MODELS = {
    "hybrid": {
        "name": "HybridGCN",
        "train_fn": train_hybrid,
        "defaults": {
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.2,
            "pooling": "attention",
        },
    },
    "edge_gat": {
        "name": "EdgeAwareGAT",
        "train_fn": train_edge_gat,
        "defaults": {
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.2,
            "pooling": "attention",
            "num_heads": 4,
        },
    },
    "sage": {
        "name": "GraphSAGE",
        "train_fn": train_sage,
        "defaults": {
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.2,
            "pooling": "attention",
            "aggregator": "mean",
        },
    },
    "gin": {
        "name": "GIN",
        "train_fn": train_gin,
        "defaults": {
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.2,
            "pooling": "add",
        },
    },
    "gated": {
        "name": "GatedGNN",
        "train_fn": train_gated,
        "defaults": {
            "hidden_dim": 256,
            "num_layers": 5,
            "dropout": 0.2,
            "pooling": "attention",
        },
    },
    "graph_transformer": {
        "name": "GraphTransformer",
        "train_fn": train_graph_transformer,
        "defaults": {
            "hidden_dim": 256,
            "num_layers": 4,
            "dropout": 0.2,
            "pooling": "both",
            "num_heads": 8,
        },
    },
    "gps": {
        "name": "GPS",
        "train_fn": train_gps,
        "defaults": {
            "hidden_dim": 256,
            "num_layers": 4,
            "dropout": 0.2,
            "pooling": "mean",
            "num_heads": 4,
        },
    },
    "pna": {
        "name": "PNA",
        "train_fn": train_pna,
        "defaults": {
            "hidden_dim": 128,  # Smaller due to high param count
            "num_layers": 3,
            "dropout": 0.2,
            "pooling": "both",
        },
    },
}


def run_experiments(
    dataset_path: str,
    models: Optional[list[str]] = None,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 5,
    min_delta: float = 0.001,
    seed: int = 42,
    output_dir: str = "results/experiments",
    verbose: bool = True,
) -> dict:
    """
    Run experiments for all specified models and collect results.

    Args:
        dataset_path: Path to dataset directory
        models: List of model keys to run (default: all)
        epochs: Max training epochs
        batch_size: Batch size
        patience: Early stopping patience
        min_delta: Minimum improvement to reset patience
        seed: Random seed
        output_dir: Directory to save experiment results
        verbose: Print progress

    Returns:
        Dictionary with all results and analytics
    """
    if models is None:
        models = list(MODELS.keys())

    # Validate model names
    invalid = [m for m in models if m not in MODELS]
    if invalid:
        raise ValueError(f"Unknown models: {invalid}. Valid: {list(MODELS.keys())}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(output_dir) / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if verbose:
        print("=" * 60)
        print("EXPERIMENT RUNNER")
        print("=" * 60)
        print(f"Dataset: {dataset_path}")
        print(f"Models: {', '.join(models)}")
        print(f"Early stopping: patience={patience}, min_delta={min_delta}")
        print(f"Output: {experiment_dir}")
        print("=" * 60)

    for model_key in models:
        model_info = MODELS[model_key]
        model_name = model_info["name"]
        train_fn = model_info["train_fn"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {model_name}")
            print("=" * 60)

        try:
            # Prepare arguments
            kwargs = {
                "dataset_path": dataset_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "patience": patience,
                "min_delta": min_delta,
                "seed": seed,
                "output_dir": str(experiment_dir / model_key),
                "verbose": verbose,
                **model_info["defaults"],
            }

            result = train_fn(**kwargs)
            all_results[model_key] = {
                "status": "success",
                "result": result,
            }

            if verbose:
                metrics = result["results"]["threshold_from_val_metrics"]
                print(f"\n{model_name} completed:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 Score: {metrics['f1']:.4f}")
                print(f"  AUC:      {metrics['auc']:.4f}")

        except Exception as e:
            if verbose:
                print(f"\nERROR training {model_name}: {e}")
            all_results[model_key] = {
                "status": "error",
                "error": str(e),
            }

    # Generate analytics
    analytics = generate_analytics(all_results, verbose=verbose)

    # Save combined results
    experiment_results = {
        "timestamp": timestamp,
        "dataset_path": dataset_path,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "min_delta": min_delta,
            "seed": seed,
        },
        "models_run": models,
        "results": all_results,
        "analytics": analytics,
    }

    with open(experiment_dir / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    # Save analytics summary as separate file for easy viewing
    with open(experiment_dir / "analytics_summary.txt", "w", encoding="utf-8") as f:
        f.write(format_analytics_report(analytics, all_results))

    if verbose:
        print(f"\nExperiment results saved to {experiment_dir}")

    return experiment_results


def generate_analytics(results: dict, verbose: bool = True) -> dict:
    """Generate analytics comparing all model results."""
    successful = {k: v for k, v in results.items() if v["status"] == "success"}

    if not successful:
        return {"error": "No successful runs to analyze"}

    # Extract metrics for comparison
    metrics_comparison = {}
    for model_key, data in successful.items():
        r = data["result"]["results"]
        val_metrics = r["threshold_from_val_metrics"]
        youden_metrics = r["youden_val_metrics"]

        metrics_comparison[model_key] = {
            "model_name": data["result"]["model"],
            "num_params": r["num_params"],
            "training_time": r["training_time"],
            "best_val_loss": r["best_val_loss"],
            # Val threshold metrics (primary)
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
            "auc": val_metrics["auc"],
            # Youden metrics (secondary)
            "youden_f1": youden_metrics["f1"],
            "youden_accuracy": youden_metrics["accuracy"],
        }

    # Compute rankings
    rankings = compute_rankings(metrics_comparison)

    # Find best models per metric
    best_models = {}
    for metric in ["accuracy", "f1", "auc", "precision", "recall"]:
        best_key = max(metrics_comparison.keys(), key=lambda k: metrics_comparison[k][metric])
        best_models[metric] = {
            "model": metrics_comparison[best_key]["model_name"],
            "value": metrics_comparison[best_key][metric],
        }

    # Compute average metrics
    avg_metrics = {}
    for metric in ["accuracy", "f1", "auc"]:
        values = [m[metric] for m in metrics_comparison.values()]
        avg_metrics[metric] = sum(values) / len(values)

    analytics = {
        "metrics_comparison": metrics_comparison,
        "rankings": rankings,
        "best_models": best_models,
        "average_metrics": avg_metrics,
        "total_models_run": len(results),
        "successful_runs": len(successful),
        "failed_runs": len(results) - len(successful),
    }

    if verbose:
        print_analytics(analytics)

    return analytics


def compute_rankings(metrics_comparison: dict) -> dict:
    """Compute model rankings for each metric."""
    rankings = {}

    for metric in ["accuracy", "f1", "auc", "training_time", "num_params"]:
        # For time and params, lower is better
        reverse = metric not in ["training_time", "num_params"]
        sorted_models = sorted(
            metrics_comparison.keys(),
            key=lambda k: metrics_comparison[k][metric],
            reverse=reverse
        )
        rankings[metric] = [
            {"rank": i + 1, "model": metrics_comparison[m]["model_name"], "value": metrics_comparison[m][metric]}
            for i, m in enumerate(sorted_models)
        ]

    return rankings


def print_analytics(analytics: dict):
    """Print formatted analytics to console."""
    print("\n" + "=" * 60)
    print("EXPERIMENT ANALYTICS")
    print("=" * 60)

    print(f"\nRuns: {analytics['successful_runs']}/{analytics['total_models_run']} successful")

    # Print comparison table
    print("\n" + "-" * 80)
    print(f"{'Model':<20} {'Acc':>8} {'F1':>8} {'AUC':>8} {'ValLoss':>8} {'Params':>10} {'Time(s)':>8}")
    print("-" * 80)

    for model_key, m in sorted(
        analytics["metrics_comparison"].items(),
        key=lambda x: x[1]["f1"],
        reverse=True
    ):
        print(
            f"{m['model_name']:<20} "
            f"{m['accuracy']:>8.4f} "
            f"{m['f1']:>8.4f} "
            f"{m['auc']:>8.4f} "
            f"{m['best_val_loss']:>8.4f} "
            f"{m['num_params']:>10,} "
            f"{m['training_time']:>8.1f}"
        )

    print("-" * 80)

    # Print best models
    print("\nBest Models:")
    for metric, info in analytics["best_models"].items():
        print(f"  {metric:>12}: {info['model']:<20} ({info['value']:.4f})")

    # Print rankings by F1 score
    print("\nRankings by F1 Score:")
    for item in analytics["rankings"]["f1"]:
        print(f"  {item['rank']}. {item['model']:<20} ({item['value']:.4f})")


def format_analytics_report(analytics: dict, results: dict) -> str:
    """Format analytics as a text report."""
    lines = []
    lines.append("=" * 60)
    lines.append("EXPERIMENT ANALYTICS REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Successful runs: {analytics['successful_runs']}/{analytics['total_models_run']}")
    lines.append("")

    # Comparison table
    lines.append("-" * 80)
    lines.append(f"{'Model':<20} {'Acc':>8} {'F1':>8} {'AUC':>8} {'ValLoss':>8} {'Params':>10} {'Time(s)':>8}")
    lines.append("-" * 80)

    for model_key, m in sorted(
        analytics["metrics_comparison"].items(),
        key=lambda x: x[1]["f1"],
        reverse=True
    ):
        lines.append(
            f"{m['model_name']:<20} "
            f"{m['accuracy']:>8.4f} "
            f"{m['f1']:>8.4f} "
            f"{m['auc']:>8.4f} "
            f"{m['best_val_loss']:>8.4f} "
            f"{m['num_params']:>10,} "
            f"{m['training_time']:>8.1f}"
        )

    lines.append("-" * 80)
    lines.append("")

    # Best models
    lines.append("BEST MODELS BY METRIC:")
    for metric, info in analytics["best_models"].items():
        lines.append(f"  {metric:>12}: {info['model']:<20} ({info['value']:.4f})")
    lines.append("")

    # Rankings
    lines.append("RANKINGS BY F1 SCORE:")
    for item in analytics["rankings"]["f1"]:
        lines.append(f"  {item['rank']}. {item['model']:<20} ({item['value']:.4f})")
    lines.append("")

    lines.append("RANKINGS BY AUC:")
    for item in analytics["rankings"]["auc"]:
        lines.append(f"  {item['rank']}. {item['model']:<20} ({item['value']:.4f})")
    lines.append("")

    lines.append("RANKINGS BY TRAINING TIME (fastest first):")
    for item in analytics["rankings"]["training_time"]:
        lines.append(f"  {item['rank']}. {item['model']:<20} ({item['value']:.1f}s)")
    lines.append("")

    # Average metrics
    lines.append("AVERAGE METRICS ACROSS ALL MODELS:")
    for metric, value in analytics["average_metrics"].items():
        lines.append(f"  {metric}: {value:.4f}")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GNN experiments and generate analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  hybrid           - HybridGCN (GCN + Attention Pooling)
  edge_gat         - EdgeAwareGAT (GAT with edge features)
  sage             - GraphSAGE
  gin              - GIN (Graph Isomorphism Network)
  gated            - GatedGNN
  graph_transformer - GraphTransformer (TransformerConv)
  gps              - GPS (GINEConv + Global Attention)
  pna              - PNA (Principal Neighborhood Aggregation) - GPU recommended

Examples:
  # Run all models
  python run_experiments.py --dataset_path /path/to/data

  # Run specific models
  python run_experiments.py --dataset_path /path/to/data --models hybrid edge_gat gps

  # Run with custom settings
  python run_experiments.py --dataset_path /path/to/data --epochs 100 --patience 3
        """
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset directory containing .pt files",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODELS.keys()),
        help="Models to run (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Min improvement to reset patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiments",
        help="Directory to save experiment results",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    run_experiments(
        dataset_path=args.dataset_path,
        models=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
