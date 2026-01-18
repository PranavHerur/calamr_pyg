"""Generic dataset loader for PyTorch Geometric graphs."""

from typing import Tuple
from torch.utils.data import Subset

from calamr_pyg.data.datasets.utils import load_dataset, split_dataset_random


def load_and_split_dataset(
    dataset_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Load dataset from directory and split into train/val/test sets.

    Finds all *.pt files in the directory, loads them, and splits them randomly.

    Args:
        dataset_dir: Path to directory containing .pt files
                     Example: "/path/to/medhallu/v8/labeled"
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_data, val_data, test_data) as Subset objects

    Example:
        >>> train, val, test = load_and_split_dataset("/path/to/data")
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    # Load all .pt files from directory
    dataset = load_dataset(dataset_dir)

    # Split into train/val/test
    ratios = [train_ratio, val_ratio, test_ratio]
    train_data, val_data, test_data = split_dataset_random(
        dataset, ratios, random_seed
    )

    return train_data, val_data, test_data
